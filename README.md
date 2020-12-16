# 2020DA_CapstoneProject

2020학년도 2학기 데이터분석 캡스톤 디자인 프로젝트 파일을 공개하기 위한 Repo입니다.


Google Drive : https://drive.google.com/drive/folders/1coYOcNSM0Zt3LNgoP1daZi7UAqkOWBvz?usp=sharing

*  Real-time Emotion Recognition\
<img src="./Demo-Image/real-time-emotion-recognition/real-time-예시.gif" width="40%"></img>


*  How to create personal emoji
1. 왼쪽부터 StyleGAN2를 통해 생성한 Emojify 이미지 원본
2. cv2의 Contour 함수로 boundary 찾아서 mask를 씌운 결과물
3. mask 부분을 투명하게 제거하고 png 파일로 저장한 결과물
4. 얼굴 중심으로 crop한 결과물

|  <img src="./Demo-Image/personal-emoji/origin.jpg" width="80%"></img>  |  <img src="./Demo-Image/personal-emoji/masked.png" width="80%"></img>  |  <img src="./Demo-Image/personal-emoji/remove-background.png" width="80%"></img>  |  <img src="./Demo-Image/personal-emoji/cropped.png" width="80%"></img>  |
|:---:|:---:|:---:|:---:|


## DataSet
*  FFHQ(pretained network) : base network로 사용.\
<img src="./Demo-Image/stylegan2/ffhq-원본.jpg" width="90%"></img>


*  Face of baby characters in Disney/Pixar\
<img src="./Demo-Image/stylegan2/baby-원본.jpg" width="90%"></img>


*  Face of characters in Disney/Pixar\
<img src="./Demo-Image/stylegan2/cartoon-원본.jpg" width="90%"></img>


* Emoji\
<img src="./Demo-Image/stylegan2/emoji-원본.jpg" width="90%"></img>


## Training Results
*  Baby-fy Results\
<img src="./Demo-Image/stylegan2/baby-fy.jpg" width="90%"></img>


*  Tooni-fy Results\
<img src="./Demo-Image/stylegan2/cartooni-fy.jpg" width="90%"></img>


*  Emoji-fy Results\
<img src="./Demo-Image/stylegan2/emoji-fy.jpg" width="90%"></img>
