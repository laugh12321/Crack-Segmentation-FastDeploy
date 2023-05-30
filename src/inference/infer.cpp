#include "infer.h"
#include "fastdeploy/vision/visualize/visualize.h"

bool fileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

CrackInfer::CrackInfer(const std::string& config_file) {
    if (!fileExists(config_file)) {
        std::cerr << "Config file " << config_file << " does not exist." << std::endl;
        return;
    }

    YAML::Node config = YAML::LoadFile(config_file);
    if (!config) {
        std::cerr << "Failed to load config file " << config_file << "." << std::endl;
        return;
    }

    std::cout << "Initializing model ...... " << std::endl;
    max_batch_size = config["max_batch_size"].as<int>();
    setRuntimeOption(config);
    model = new fastdeploy::vision::segmentation::PaddleSegModel(
        config["model_dir"].as<std::string>() + sep + "model.pdmodel",
        config["model_dir"].as<std::string>() + sep + "model.pdiparams",
        config["model_dir"].as<std::string>() + sep + "deploy.yaml",
        option);

    assert(model->Initialized());
    std::cout << "Successfully initialized the model!" << std::endl;
}


void CrackInfer::setRuntimeOption(const YAML::Node& config) {
    option.UseGpu(config["gpu_id"].as<int>());
    std::string backend = config["backend"].as<std::string>();
    if (backend.find("Paddle") != std::string::npos) {
        option.UsePaddleInferBackend();
        if (backend.find("TRT") != std::string::npos) {
            option.paddle_infer_option.enable_trt = true;
        }
    } else if (backend.find("TensorRT") != std::string::npos) {
        option.UseTrtBackend();
        option.trt_option.enable_fp16 = config["enable_fp16"].as<bool>();
        option.trt_option.serialize_file = config["model_dir"].as<std::string>() + sep + "model.trt";
        for (auto shape : config["set_shape"]) {
            option.trt_option.SetShape(
                shape["tensor_name"].as<std::string>(), 
                shape["min"].as<std::vector<int32_t>>(), 
                shape["opt"].as<std::vector<int32_t>>(), 
                shape["max"].as<std::vector<int32_t>>()
            );
        }
    } else if (backend.find("ONNX") != std::string::npos) {
        option.UseOrtBackend();
    }
}


std::vector<std::pair<int, cv::Mat>> CrackInfer::batchinfer(const std::vector<std::string>& images_path) {
    std::cout << "Predicting ...... " << std::endl;

    std::vector<cv::Mat> images;
    std::vector<std::pair<int, cv::Mat>> predicts;
    std::vector<fastdeploy::vision::SegmentationResult> results;

    // Split images into batches
    std::vector<std::vector<std::string>> image_batches;
    size_t num_images = images_path.size();
    size_t num_batches = (num_images + max_batch_size - 1) / max_batch_size;

    for (size_t i = 0; i < num_batches; ++i) {
        size_t start = i * max_batch_size;
        size_t end = std::min(start + max_batch_size, num_images);
        image_batches.push_back(std::vector<std::string>(images_path.begin() + start, images_path.begin() + end));
    }

    // Predict for each batch
    for (const auto& image_batch : image_batches) {
        // Read images
        images.clear();
        for (const std::string& image_path : image_batch) {
            cv::Mat image = cv::imread(image_path);
            images.push_back(image);
        }

        // Predict
        results.clear();
        if (!model->BatchPredict(images, &results)) {
            std::cerr << "Failed to predict." << std::endl;
            return {};
        }

        // Save results
        for (size_t idx = 0; idx < image_batch.size(); ++idx) {
            int label = 0;
            cv::Mat vis_image = fastdeploy::vision::VisSegmentation(images[idx], results[idx]);
            std::set<int> labels(results[idx].label_map.begin(), results[idx].label_map.end());
            if (labels.size() != 1) {
                label = 1;
            }

            predicts.push_back(std::make_pair(label, vis_image));
        }
    }

    std::cout << "Successfully predicted!" << std::endl;
    return predicts;
}
