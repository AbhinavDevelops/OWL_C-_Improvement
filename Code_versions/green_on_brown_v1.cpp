#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>

namespace fs = std::filesystem;

cv::Mat exg(const cv::Mat &image)
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    cv::Mat b = channels[0], g = channels[1], r = channels[2];
    b.convertTo(b, CV_32F);
    g.convertTo(g, CV_32F);
    r.convertTo(r, CV_32F);
    cv::Mat out = 2 * g - r - b;
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

    int exg_min = 30, exg_max = 250, min_detection_area = 100;
    bool show_display = false;

    int image_count = 0;
    double total_time = 0.0;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});

    for (const auto &entry : fs::directory_iterator(dataset_path))
    {
        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
            continue;

        auto start = std::chrono::high_resolution_clock::now();

        // --- Algorithm logic ---
        cv::Mat exg_img = exg(image);
        cv::Mat clipped;
        cv::threshold(exg_img, clipped, exg_min, 255, cv::THRESH_TOZERO);
        cv::threshold(clipped, clipped, exg_max, 255, cv::THRESH_TRUNC);

        cv::Mat binary;
        cv::adaptiveThreshold(clipped, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 31, 2);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat output = image.clone();
        int weed_count = 0;
        for (const auto &c : contours)
        {
            if (cv::contourArea(c) > min_detection_area)
            {
                cv::Rect rect = cv::boundingRect(c);
                cv::rectangle(output, rect, cv::Scalar(0, 0, 255), 2);
                cv::putText(output, "WEED", {rect.x, rect.y + 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            {255, 0, 0}, 2);
                ++weed_count;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += elapsed;
        ++image_count;

        std::string out_path = output_path + "output_C++" + entry.path().filename().string();
        cv::imwrite(out_path, output);

        std::cout << "Processed " << entry.path().filename()
                  << " | Weeds: " << weed_count
                  << " | Time: " << elapsed << " ms" << std::endl;

        if (show_display)
        {
            cv::imshow("Result", output);
            cv::waitKey(0);
        }
    }

    std::cout << "Average inference time: " << (total_time / image_count) << " ms\n";
    return 0;
}
