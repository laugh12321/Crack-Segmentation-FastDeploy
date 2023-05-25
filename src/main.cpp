#include "inference/infer.h"
#include <iostream>
#include <string>

int main() {

	std::string config_file = "E:/Laugh/Projects/crackinfer/assets/config.yml";

    CrackInfer crackinfer = CrackInfer(config_file);

    std::vector<std::string> images_path{"E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg"};
    std::string save_dir = "E:/Laugh/Pictures";

    std::vector<fastdeploy::vision::SegmentationResult> results = crackinfer.batchinfer(images_path, save_dir);
    for (auto& result : results) {
        std::cout << result.Str() << std::endl;
    }
}