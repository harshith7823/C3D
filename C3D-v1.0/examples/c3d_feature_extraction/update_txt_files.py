import cv2
import math
import os

def update_input_output_list_video_txt(video,video_n):
	video = cv2.VideoCapture(video)
	video_n1 = video_n[0:-4]
	extsn = video_n[-4:]
	frame_count =video.get(cv2.CAP_PROP_FRAME_COUNT)
	frame_categories = (math.floor(frame_count/16))
	res1 = ""
	res2 = ""
	for i in range(frame_categories):
		res1 += "input/avi/"+video_n1+extsn+" "+str(16*i)+" 0"
		res1 += "\n"
		res2 += "output/c3d/"+video_n+"/"
		frame_str = str(16*i)
		ini = ""
		for k in range(6 - len(frame_str)):
			ini += "0"
		res2 += ini + frame_str
		res2 += "\n"

	with open('./prototxt/input_list_video.txt', 'a') as the_file:
		the_file.write(res1)

	with open('./prototxt/output_list_video_prefix.txt', 'a') as the_file:
		the_file.write(res2)

def update_sh_script(video):	
	with open('./c3d_sport1m_feature_extraction_video.sh', 'a') as the_file:
		the_file.write("mkdir -p output/c3d/"+video+"\n")

with open('./c3d_sport1m_feature_extraction_video.sh', 'w') as the_file:
	the_file.write("")

with open('./prototxt/input_list_video.txt', 'w') as the_file:
	the_file.write("")

with open('./prototxt/output_list_video_prefix.txt', 'w') as the_file:
	the_file.write("")
		
p = "./input/avi"
for video in sorted(os.listdir(p)):
  if video == ".ipynb_checkpoints":
    continue
  update_input_output_list_video_txt(p+"/" + video,video)
  update_sh_script(video)

number_of_batches = 0
with open('./prototxt/input_list_video.txt', 'r') as the_file:
	number_of_batches = str(math.ceil(len(the_file.readlines())/50))

with open('./c3d_sport1m_feature_extraction_video.sh', 'a') as the_file:
	the_file.write("GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000 0 50 "+ number_of_batches +" prototxt/output_list_video_prefix.txt fc6-1")

