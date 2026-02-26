#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ctime>
#include <iostream>
#include <fstream>

int main() {
    cv::VideoCapture cap("http://192.168.0.50:8082/stream"); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot access the camera\n";
        return -1;
    }

    // yolo image import
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(
        "yolov4-tiny.cfg",
        "yolov4-tiny.weights"
    );

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<std::string> output_layers;
    for (int id : net.getUnconnectedOutLayers()) {
        output_layers.push_back(net.getLayerNames()[id - 1]);
    }

    bool recording = false;
    bool recording_restart = true;

    // used to calculate different in time
    std::time_t record_start_time = 0;
    std::time_t last_record_end = 0;

    // total time of recording
    const int RECORD_SECONDS = 60;
    // time between end of recording and next detection claim
    const int COOLDOWN_SECONDS = 5;

    cv::VideoWriter writer;
    cv::Size output_size(1280, 720);

    cv::Mat frame;

    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty()) break;

        cv::Mat resized;
        cv::resize(frame, resized, output_size);

        /*
            ######################
            YOLO DETECTION START
            #######################
        */
        cv::Mat blob = cv::dnn::blobFromImage(
            resized, 1 / 255.0, cv::Size(416, 416),
            cv::Scalar(), true, false
        );

        net.setInput(blob);
        std::vector<cv::Mat> detections;
        net.forward(detections, output_layers);

        bool person_detected = false;

        for (const auto& output : detections) {
            for (int i = 0; i < output.rows; i++) {
                const float* data = output.ptr<float>(i);
                float confidence = data[4];

                if (confidence > 0.5f) {
                    int class_id = std::max_element(data + 5, data + output.cols) - (data + 5);
                    float score = data[5 + class_id];

                    // COCO class 0 = person
                    if (class_id == 0 && score > 0.5f) {
                        person_detected = true;

                        int cx = int(data[0] * resized.cols);
                        int cy = int(data[1] * resized.rows);
                        int w  = int(data[2] * resized.cols);
                        int h  = int(data[3] * resized.rows);

                        cv::Rect box(cx - w / 2, cy - h / 2, w, h);
                        cv::rectangle(resized, box, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        /*
            ######################
            YOLO DETECTION END
            #######################
        */

        // if person detected activate variables to start recording and file storage
        if (person_detected && recording_restart && !recording) {
            recording = true;
            recording_restart = false;
            record_start_time = std::time(nullptr);

            char filename[100];

            // Format: YYYY-MM-DD HH:MM:SS
            std::strftime(filename, sizeof(filename), "vid_cap_%Y-%m-%d_%H-%M-%S.mp4", std::localtime(&record_start_time));

            writer.open(filename,
                        cv::VideoWriter::fourcc('a','v','c','1'), // H.264 codec for MP4
                        20,
                        output_size);

            if (!writer.isOpened()) {
                std::cerr << "Error: Cannot open video writer\n";
                return -1;
            }

            std::cout << "Recording started: " << filename << "\n";
        }

        // start importing frames to the file for 60 sec
        if (recording) {
            writer.write(resized);

            if ( (std::time(nullptr) - record_start_time) >= RECORD_SECONDS) {
                recording = false;
                last_record_end = std::time(nullptr);
                writer.release();
                std::cout << "Recording stopped\n";
            }
        }

        // detection cooldown
        if (!recording_restart &&
            !recording &&
            (std::time(nullptr) - last_record_end) >= COOLDOWN_SECONDS) {
            recording_restart = true;
            std::cout << "System Recording Ready\n";
        }

        cv::imshow("YOLO Human Detection", resized); // Display the frame
        if (cv::waitKey(30) >= 0) break; // Exit on key press
    }

    if (writer.isOpened()) writer.release();
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
