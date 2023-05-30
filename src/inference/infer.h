#ifndef _INFER_H_
#define _INFER_H_

#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/vision.h"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

bool fileExists(const std::string& filePath);

class CrackInfer {
public:
    CrackInfer(const std::string& config_file);
    std::vector<std::pair<int, cv::Mat>> batchinfer(const std::vector<std::string> &images_path);
private:
    fastdeploy::vision::segmentation::PaddleSegModel* model;
    fastdeploy::RuntimeOption option;
    int max_batch_size;
    void setRuntimeOption(const YAML::Node& config);
};

#endif // _INFER_H_