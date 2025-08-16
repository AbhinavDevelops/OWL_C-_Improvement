// green_on_brown_video.cpp
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>

// ──────────────────────────── helpers ─────────────────────────────
cv::Mat clip(const cv::Mat &src, int minVal, int maxVal)
{
    cv::Mat tmp, dst;
    cv::max(src, minVal, tmp);
    cv::min(tmp, maxVal, dst);
    return dst;
}

cv::Mat exg(const cv::Mat &image)
{
    std::vector<cv::Mat> ch;
    cv::split(image, ch); // B G R
    cv::Mat b = ch[0], g = ch[1], r = ch[2];
    b.convertTo(b, CV_32F);
    g.convertTo(g, CV_32F);
    r.convertTo(r, CV_32F);
    cv::Mat out = 2 * g - r - b; // ExG
    cv::threshold(out, out, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(out, out, 0, 0, cv::THRESH_TOZERO);
    out.convertTo(out, CV_8U);
    return out;
}

// ─────────────────────────── benchmark ───────────────────────────
int main(int argc, char **argv)
{
    /* ───── configuration ───── */
    std::string videoSrc = (argc > 1) ? argv[1] : "0"; // “0” = default cam
    bool saveVideo = true;                             // set true to write MP4
    const std::string outFile = "../datasets/videos_output/gob_bench.mp4";

    const int exg_min = 30, exg_max = 250;
    const int min_detection_area = 100;
    const int warmupFrames = 30; // skip timing first N frames
    const int maxFrames = 300;   // set 0 ⇒ run until EOF

    /* ───── open capture ───── */
    cv::VideoCapture cap;
    if (videoSrc == "0")
        cap.open(0, cv::CAP_ANY);
    else
        cap.open(videoSrc, cv::CAP_ANY);

    if (!cap.isOpened())
    {
        std::cerr << "❌  Could not open video source: " << videoSrc << '\n';
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fpsIn = cap.get(cv::CAP_PROP_FPS);
    if (fpsIn <= 0)
        fpsIn = 30; // fallback for some webcams

    cv::VideoWriter writer;
    if (saveVideo)
    {
        int fourcc = CV_FOURCC('m', 'p', '4', 'v');
        writer.open(outFile, fourcc, fpsIn, {width, height});
        if (!writer.isOpened())
            std::cerr << "⚠️  Could not open writer; continuing without video out\n";
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});

    /* ───── benchmark variables ───── */
    double algoTimeMs = 0.0;
    int frameCount = 0;

    // ─── warm‑up ───
    for (int i = 0; i < warmupFrames; ++i)
    {
        cv::Mat f;
        if (!cap.read(f))
            break;
        (void)exg(f); // run once just to load caches
    }

    auto wallStart = std::chrono::steady_clock::now();

    // ─── timed loop ───
    while (true)
    {
        if (maxFrames && frameCount >= maxFrames)
            break;

        cv::Mat frame;
        if (!cap.read(frame))
            break; // EOF or camera closed

        auto t0 = std::chrono::steady_clock::now();

        // 1) ExG
        cv::Mat exgImg = exg(frame);

        // 2) clip to [exg_min, exg_max]
        cv::Mat clipped = clip(exgImg, exg_min, exg_max);

        // 3) adaptiveThreshold
        cv::Mat bin;
        cv::adaptiveThreshold(clipped, bin, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, 2);

        // 4) morphology
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, {-1, -1}, 1);

        // 5) contours & annotation
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int weeds = 0;
        for (auto &c : contours)
        {
            if (cv::contourArea(c) > min_detection_area)
            {
                cv::Rect r = cv::boundingRect(c);
                cv::rectangle(frame, r, {0, 0, 255}, 2);
                cv::putText(frame, "WEED", {r.x, r.y + 30},
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, {255, 0, 0}, 2);
                ++weeds;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        algoTimeMs += ms;
        ++frameCount;
        double fps_frame = 1000.0 / ms;
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps_frame));
        cv::putText(frame, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

        std::cout << "\rFrame " << frameCount
                  << " | weeds: " << weeds
                  << " | " << ms << " ms           " << std::flush;

        if (saveVideo && writer.isOpened())
            writer.write(frame);
        //  optional live view (commented to keep timing clean)
        // cv::imshow("GOB Video", frame);
        //  if (cv::waitKey(1) == 27) break;   // ESC
    }

    auto wallEnd = std::chrono::steady_clock::now();
    double wallS = std::chrono::duration<double>(wallEnd - wallStart).count();

    cap.release();
    if (writer.isOpened())
        writer.release();
    cv::destroyAllWindows();

    /* ───── report ───── */
    double meanAlgoMs = algoTimeMs / frameCount;
    double fpsAlgo = 1000.0 / meanAlgoMs;
    std::cout << "\n────────────────────────── RESULTS ──────────────────────────\n";
    std::cout << "Frames processed : " << frameCount << '\n';
    std::cout << "Mean per‑frame   : " << meanAlgoMs << " ms\n";
    std::cout << "Mean FPS (algo)  : " << fpsAlgo << '\n';
    std::cout << "Wall‑clock time  : " << wallS << " s (capture + I/O)\n";
    std::cout << "──────────────────────────────────────────────────────────────\n";
    return 0;
}
