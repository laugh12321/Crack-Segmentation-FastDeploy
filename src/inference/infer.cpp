#include "infer.h"
#include "fastdeploy/vision/visualize/visualize.h"

CrackInfer::CrackInfer(const std::string& config_file) {
    YAML::Node config = YAML::LoadFile(config_file);
    std::cout << "Initializing model ...... " << std::endl;
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

std::vector<fastdeploy::vision::SegmentationResult> CrackInfer::batchinfer(const std::vector<std::string> &images_path, const std::string &save_dir) {
    std::cout << "Predicting ...... " << std::endl;

    std::vector<fastdeploy::vision::SegmentationResult> results;
    std::vector<cv::Mat> images;

    // read images
    for (const std::string& image_path : images_path) {
        cv::Mat image = cv::imread(image_path);
        images.push_back(image);
    }

    // predict
    if (!model->BatchPredict(images, &results)) {
        std::cerr << "Failed to predict." << std::endl;
        return {};
    }

    // save results
    for (size_t idx = 0; idx < images_path.size(); idx++) {
        // get filename
        std::string::size_type iPos = images_path[idx].find_last_of(sep) + 1;
        std::string filename = images_path[idx].substr(iPos, images_path[idx].length() - iPos);

        cv::Mat vis_image = fastdeploy::vision::VisSegmentation(images[idx], results[idx]);
        cv::imwrite(save_dir + sep + filename, vis_image);
    }

    std::cout << "Successfully predicted!" << std::endl;
    return results;
}