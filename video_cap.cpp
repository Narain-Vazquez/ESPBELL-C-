#include <opencv2/opencv.hpp>
#include <ctime>

int main() {
    cv::VideoCapture cap("http://192.168.0.50:8082/stream"); // Open the default camera
    if (!cap.isOpened()) {
        printf("Error: Cannot access the camera\n");
        return -1;
    }

    cv::Mat frame, resized_frame;

    std::time_t now = time(NULL); // getting current time
    char buffer[100];

    // Format: YYYY-MM-DD HH:MM:SS
    std::strftime(buffer, sizeof(buffer), "vid_cap_%Y-%m-%d_%H-%M-%S.mp4", std::localtime(&now));
    cv::Size output_size(1280, 720);

    cv::VideoWriter writer(
        buffer,
        cv::VideoWriter::fourcc('a','v','c','1'), // H.264 codec for MP4
        20,
        output_size
    );

    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open video writer\n";
        return -1;
    }

    while (time(NULL) - now < 15) // record for 15 sec
    {
        cap >> frame; // Capture a frame
        if (frame.empty()) break;

        
        cv::resize(frame, resized_frame, cv::Size(1280, 720)); 
        writer.write(resized_frame);

        cv::imshow("Captured Frame", resized_frame); // Display the frame
        if (cv::waitKey(30) >= 0) break; // Exit on key press
    }

    cap.release(); // Release the camera
    writer.release(); // Release the video converter
    cv::destroyAllWindows();  // just in case no memoery leaks

    std::cout << "Video saved as: " << buffer << "\n";

    return 0;
}
