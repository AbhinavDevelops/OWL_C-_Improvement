#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat exg(cv::Mat image)
{
    std::vector<cv::Mat>
        channels;
    cv::split(image, channels);

    cv::Mat blue = channels[0];
    cv::Mat green = channels[1];
    cv::Mat red = channels[2];

    blue.convertTo(blue, CV_32F);
    green.convertTo(green, CV_32F);
    red.convertTo(red, CV_32F);

    cv::Mat image_out = 2 * green - red - blue;

    // Clip the values to the range [0, 255]
    cv::threshold(image_out, image_out, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(image_out, image_out, 0, 0, cv::THRESH_TOZERO);

    // Convert back to 8-bit for display
    image_out.convertTo(image_out, CV_8U);

    return image_out;
}

int main(int argc, char **argv)
{
    // Read the image file
    cv::Mat image = cv::imread("../datasets/my_dataset/2de33465be3ab7e7363a.jpg");

    // Check if the image was successfully loaded
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Display the image in a window
    cv::Mat exg_image = exg(image);

    // Wait for a key press indefinitely
    cv::imshow("ExG Output", exg_image);
    cv::waitKey(0);

    return 0;
}
