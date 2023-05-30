#include "inference/infer.h"
#include <iostream>
#include <string>
#include <set>

int main() {

	std::string config_file = "E:/Laugh/Projects/Crack-Segmentation-FastDeploy/assets/config.yml";

    CrackInfer crackinfer = CrackInfer(config_file);

    std::vector<std::string> images_path{"E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg", "E:/Laugh/Projects/yolov5/runs/camera/image/back/13/04/2023_04_07_13_04_00.389.jpg"};

    std::cout << "images_path.size: " << images_path.size() << std::endl;
    std::vector<std::pair<int, cv::Mat>> results = crackinfer.batchinfer(images_path);
    for (auto& result : results) {
        std::cout << "crack label: ";
        std::cout << result.first << std::endl;
    }
}