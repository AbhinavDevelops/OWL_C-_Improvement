#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>

namespace fs = std::filesystem;

// exactly the Python: np.clip(output, min, max)
cv::Mat clip(const cv::Mat &src, int minVal, int maxVal)
{
    cv::Mat tmp, dst;
    cv::max(src, minVal, tmp);
    cv::min(tmp, maxVal, dst);
    return dst;
}

// your original ExG implementation
cv::Mat exg(const cv::Mat &image)
{
    std::vector<cv::Mat> ch;
    cv::split(image, ch);
    cv::Mat b = ch[0], g = ch[1], r = ch[2];
    b.convertTo(b, CV_32F);
    g.convertTo(g, CV_32F);
    r.convertTo(r, CV_32F);
    cv::Mat out = 2 * g - r - b;
    // clip to [0,255]
    cv::threshold(out, out, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(out, out, 0, 0, cv::THRESH_TOZERO);
    out.convertTo(out, CV_8U);
    return out;
}

int main()
{
    std::string dataset_path = "../datasets/my_dataset/";
    std::string output_path = "../datasets/output/";
    fs::create_directories(output_path);

    int exg_min = 30;
    int exg_max = 250;
    int min_detection_area = 100;
    bool show_display = false;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {3, 3});

    int image_count = 0;
    double total_time = 0.0;

    for (auto &entry : fs::directory_iterator(dataset_path))
    {
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty())
            continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 1) ExG
        cv::Mat exg_img = exg(img);

        // 2) clip to [exg_min, exg_max]
        cv::Mat clipped = clip(exg_img, exg_min, exg_max);

        // 3) adaptiveThreshold (Gaussian), binary inverted
        cv::Mat thresh;
        cv::adaptiveThreshold(
            clipped, thresh,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            /*blockSize*/ 31,
            /*C*/ 2);

        // 4) morphological close, 1 iteration
        cv::morphologyEx(
            thresh, thresh,
            cv::MORPH_CLOSE, kernel,
            /*anchor*/ cv::Point(-1, -1),
            /*iterations*/ 1);

        // 5) find & filter contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            thresh, contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        cv::Mat out = img.clone();
        int weed_count = 0;
        for (auto &c : contours)
        {
            if (cv::contourArea(c) > min_detection_area)
            {
                cv::Rect r = cv::boundingRect(c);
                cv::rectangle(out, r, {0, 0, 255}, 2);
                cv::putText(
                    out, "WEED",
                    {r.x, r.y + 30},
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0, {255, 0, 0}, 2);
                ++weed_count;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_time += ms;
        ++image_count;

        std::string name = entry.path().filename().string();
        std::cout << "Processed " << name
                  << " | Weeds: " << weed_count
                  << " | Time: " << ms << " ms\n";

        cv::imwrite(output_path + "cpp_" + name, out);

        if (show_display)
        {
            cv::imshow("Result", out);
            cv::waitKey(0);
        }
    }

    std::cout << "Average inference time: "
              << (total_time / image_count) << " ms\n";
    return 0;
}
