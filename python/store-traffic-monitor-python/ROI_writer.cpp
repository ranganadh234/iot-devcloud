#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <vector>
#include <algorithm> 
#include <assert.h>

using namespace std;
using namespace cv;


void placeBoxes(Mat& frame, vector<string> obj, bool is_async_mode = true){
	string id = obj[0];    // frame id
	int xmin = std::stoi(obj[1]);  // xmin
	int ymin = std::stoi(obj[2]);  // ymin
	int xmax = std::stoi(obj[3]);  // xmax
	int ymax = std::stoi(obj[4]);  // ymax
	string det_label = obj[5]; //class_id
//	int class_id = std::stoi(obj[5]); //class_id
//	string prob = obj[6];   // probability
	string current_count = obj[6];
	string total_count = obj[7];	
	string det_time = obj[8]; //inference engine detection time

/*	string id = obj[0];
	float face_infer_time = std::stof(obj[1]);
	float head_pose_infer_time = std::stof(obj[2]);	
	int shopper = std::stoi(obj[3]);
	int looker = std::stoi(obj[4]);*/
	
    	Scalar color(20, 20, 20);
	String inf_time_message;
        String async_mode_message;
        if (is_async_mode){
		inf_time_message = "Inference time: N/A for async mode";
		// inf_time_message = "Face Inference time: " + std::to_string(face_infer_time) + " ms.";
		async_mode_message = "Async mode is on. Processing request: "+id;
	}
        else{
 		inf_time_message = "Inference time:"+ det_time + " ms";
		async_mode_message = "Async mode is off. Processing request: "+id;
	}
	string current_count_label = "Current " + det_label + " count: " + current_count;
	string total_count_label = "Total " + det_label + " count: " + total_count;
        rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), color, 2 );
	//putText(frame, det_label+' '+prob+'%', Point(xmin, ymin-7), CV_FONT_HERSHEY_COMPLEX, 1.0, color, 2);
        putText(frame, inf_time_message, Point(15, 15), CV_FONT_HERSHEY_COMPLEX, 0.5, color, 1.5);
        putText(frame, async_mode_message, Point(15, 30), CV_FONT_HERSHEY_COMPLEX, 0.5, color, 1.5);
	if (current_count != "N/A") {
		putText(frame, current_count_label, Point(15, 45), CV_FONT_HERSHEY_COMPLEX, 0.5, color, 1.5);
		putText(frame, total_count_label, Point(15, 60), CV_FONT_HERSHEY_COMPLEX, 0.5, color, 1.5);
	}
/*	string head_time_message = "Head pose Inference time: " + std::to_string(head_pose_infer_time) + " ms.";
	inf_time_message = "Face Inference time: " + std::to_string(face_infer_time) + " ms.";
	string shopper_s = "Shopper: " + std::to_string(shopper);
	string looker_s = "Looker: " + std::to_string(looker);

	if (head_pose_infer_time > 0.0) {
		putText(frame, head_time_message, Point(0, 35),
                        FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1);
	}
	putText(frame, inf_time_message, Point(0, 15), FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1);
        putText(frame, shopper_s, Point(0, 90), FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1);
        putText(frame, looker_s, Point(0, 110), FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1);*/
	return; 
}

string trim(const string& str)
{
    size_t first = str.find_first_not_of(' ');
    if (string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

vector<pair<string,string>> read_conf(string& filename) {
    std::vector<pair<std::string,std::string>> videoPaths;
    std::ifstream file;
    std::string word1, word2;
    file.open(filename.c_str());
    bool nextTrue = false;
    //while (file >> word)
    while (getline(file, word1, ' ')) {
	word1 = trim(word1);
	//videoPaths.push_back(word1);
	getline(file, word2, '\n');
	word2 = trim(word2);
	//if (word1 != "" and word2 != "")
	videoPaths.push_back(make_pair(word1, word2));
	//std::cout << word1 << " " << word2 << std::endl;
	//word1 = word2;
        /*word = trim(word);

		//std::cout << word << std::endl;
	std::cout << word.substr(0, word.find(' ')) << std::endl;
        if (word.substr(0, word.find(' ')) == "video") {  // 5
                nextTrue = true;
                continue;
        }
        if (nextTrue) {
                videoPaths.push_back(word);
                nextTrue = false;
        }*/
    }
    return videoPaths;
}

int main(int argc, char ** argv)
{	
	//assert(argc >= 3);
	double t = omp_get_wtime();	
	//Parse input arguments: 
	//argv[1]:(string)input_stream->path to streaming video, 
	//argv[2]:(string)result_directory->path to input data file and the output processed mp4 video
	//argv[3]:(int)(1 or 2)Skip_frame-> 2 to skip frame in the output, 1 to process the complete video without skipping
	//argv[4]:(float)resl-> scale ratio of the output frame (0.75, 0.5, 1)
	
	string job_id = getenv("PBS_JOBID");
// 	string job_id = "__og_3__";

	string config_file = argv[1];
	vector<pair<string,string>> video_paths = read_conf(config_file);
	//std::cout << "video_paths.size(): " << video_paths.size() << "\n";

	/*for (int i = 0; i < video_paths.size(); i++) {
		std::cout << i << ": " << video_paths[i].first << std::endl;
		std::cout << i << ": " << video_paths[i].second << std::endl;

		VideoCapture cap(video_paths[i].first);
		if (not cap.isOpened()) {
			std::cout << "Cannot open file" << "\n";
		} else {
			std::cout << "File is OK." << "\n";
		}
	}*/

	//string input_stream = argv[1];
	//string input_data = string(argv[2])+"/output_"+job_id+".txt";
	string progress_data = string(argv[2])+"/post_progress_"+job_id+".txt";
	//string output_result = string(argv[2])+"/output_"+job_id+".mp4";
	int skip_frame = stoi(argv[3]);
	float resl = stof(argv[4]);

// 	VideoCapture cap(video_paths[0].first);
// 	int fps = (int)cap.get(CAP_PROP_FPS);
    int total_frames = 0;
	for (int i = 0; i < video_paths.size(); i++) {
		VideoCapture cap(video_paths[i].first);
		total_frames += (int)(cap.get(CAP_PROP_FRAME_COUNT));
	}
	
	//std::cout << "fps: " << fps << "\n";

	//Start VideoCapture
    int seq_num_total = 0;
	for (int i = 0, j = video_paths.size()-1; i < video_paths.size(); i++, j--) {	
		Mat frame;
		//std::cout << video_paths[j].first << std::endl;	
  		std::string s = video_paths[i].first;
		std::string delimiter = "/";
		size_t pos = 0;

		while ((pos = s.find(delimiter)) != std::string::npos) {
    			s.erase(0, pos + delimiter.length());
		}
		//std::cout << string(argv[2])+"/output_"+job_id+"_"+s+".txt" << std::endl;

		//string input_data =  string(argv[2])+"/output_"+job_id+"_"+std::to_string(i)+".txt";
		string input_data = string(argv[2])+"/output_"+job_id+"_"+s+".txt";
		string  output_result = string(argv[2])+"/inference_output_Video_"+std::to_string(i)+".mp4"; 
// 		string  output_result = string(argv[2])+"/output_"+job_id+"_"+s;
		
		//std::cout << video_paths[i].first << " " << input_data << " ";
		
		VideoCapture cap(video_paths[i].first); 
		//std::cout << "length of video: " << cap.get(CAP_PROP_FRAME_COUNT) << std::endl;
		if(!cap.isOpened()){
			   cout << "Error opening video stream or file" << endl;
			   return -1;  
        }


		//Open the input data file and read the first line to str
		ifstream input(input_data);
		ofstream progress;
		string str;
		getline(input, str,  input.widen('\n'));
			vector<string> object(10, "0");
		int width = 0;
		int height = 0;
		int length = 0;
        int id = 0;
		int seq_num = 0;
		int next_id = 0;
		//Open the output file to write processed video to it
		VideoWriter outVideo;
		if(cap.isOpened()){
			width = int(cap.get(CAP_PROP_FRAME_WIDTH));	
			height = int(cap.get(CAP_PROP_FRAME_HEIGHT));	
			length = int(cap.get(CAP_PROP_FRAME_COUNT));
			//outVideo.open(output_result, VideoWriter::fourcc('a', 'v', 'c', '1'), fps/skip_frame, Size(width*resl, height*resl), true);
			outVideo.open(output_result, VideoWriter::fourcc('a', 'v', 'c', '1'), int(cap.get(CAP_PROP_FPS))/skip_frame, Size(width*resl, height*resl), true);
		}
		//Start while loop to process input stream and write the output frame to output_results 
			while(cap.isOpened()/* and !str.empty()*/){
			cap >> frame;
			if (frame.empty())
					break;
			while ( !str.empty()   && seq_num == id && id == next_id){
					int len = 0;
				int j = 0;
				while(len < str.size()){
					object[j].clear();
					while( len < str.size() && str[len] != ' '  && str[len] != '\n' ){
						object[j].push_back(str[len]);
						len++;
					}	
					j++;
					len++;
				}
				next_id = std::stoi(object[0]);
				if (id == next_id){
					placeBoxes(frame, object);
					getline(input, str,  input.widen('\n'));
				}
				else {
					id = next_id;
					break;
				}
			}	
			seq_num++;
            seq_num_total++;
			if (seq_num%10 == 0){
				double fps_t = omp_get_wtime()-t;
                progress.open(progress_data);
// 				string cur_progress = to_string(int(100*seq_num/length))+'\n';
// 				string remaining_time = to_string(int((fps_t/seq_num)*(length-seq_num)))+'\n';
// 				string estimated_time = to_string(int((fps_t/seq_num)*length))+'\n';
				string cur_progress = to_string(int(100*seq_num_total/total_frames))+'\n';
				string remaining_time = to_string(int((fps_t/seq_num_total)*(total_frames-seq_num_total)))+'\n';
				string estimated_time = to_string(int((fps_t/seq_num_total)*total_frames))+'\n';
				progress<<cur_progress;
				progress<<remaining_time;
				progress<<estimated_time;
				progress.flush();
				progress.close();
			}
			if (id%skip_frame == 0){
				resize(frame, frame, Size(width * resl, height * resl), 0, 0, CV_INTER_LINEAR);
				outVideo.write(frame);
			}
		}
		cap.release();
		destroyAllWindows();
	}

	t = omp_get_wtime()-t;
	cout<<"Video post-processing time: "<<t<<" seconds"<<endl;
}
