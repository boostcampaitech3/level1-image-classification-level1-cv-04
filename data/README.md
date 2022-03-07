# Naver boostcamp Level 1 P Stage post-competition dataset

* 원 대회: [**마스크 착용 상태 분류**](https://stages.ai/competitions/104/overview/description)
  * 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
  * #부스트캠프3기
* 원 대회 기간: 2022.02.23 ~ 2022.03.03 19:00
* 원 대회 규모: 참가팀 약 50여 팀, 약 250명



* post-competition 데이터셋 링크: [mega.nz](https://mega.nz/folder/WtVj0CLR#eZ4CZX5fG_lgECFgYtu7BQ)



## 원 대회 정보

### 대회 개요

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

### 원 대회 데이터 저작권

대회에 사용되는 [마스크 분류 데이터셋](https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000102/data/train.tar.gz)은 **캠프 교육용 라이선스** 아래 사용 가능하며, 교육 외부에 노출될 수 없습니다.

대회가 끝나고 나서도 후속 작업을 진행할 수 있도록, 오픈소스 데이터셋 및 라이브러리를 이용하여 *새롭게 제작된 마스크 데이터셋* 을 공개합니다.



## 데이터셋 정보

### 원 대회 데이터 개요

마스크를 착용하는 건 COIVD-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다.

### 새로운 데이터셋 제작 방향성

* 해당 데이터셋은 [Generated Media, Inc.](https://generated.photos/)의 [**100k Faces 데이터셋**](https://drive.google.com/drive/folders/1wSy4TVjSvtXeRQ6Zr8W98YbSuZXrZrgY), [NVIDIA labs](https://github.com/NVlabs)의 [**Flickr-Faces-HQ 데이터셋**](https://github.com/NVlabs/ffhq-dataset)과 [Sefik I. Serengil](https://github.com/serengil)의 [**deepface**](https://github.com/serengil/deepface) 라이브러리, [Aqeel Anwar](https://github.com/aqeelanwar)의 [**MaskTheFace**](https://sites.google.com/view/masktheface) 라이브러리를 이용하여 만들어졌습니다.
  * **100k Faces 데이터셋** : GAN으로 생성된, 세상에 존재하지 않는 사람들의 얼굴 사진 데이터셋입니다. 다양한 나이와 인종으로 구성되어 있으며, 출처([Generated Photos](https://generated.photos/))를 밝힐 시 Personal Use가 가능합니다. Commercial Use는 금지되어 있으며, 기계학습 데이터 목적으로만 사용하시길 바랍니다. ([LICENSE 전문](https://generated.photos/terms-and-conditions))
  * **Flickr-Faces-HQ 데이터셋** : Flickr의 사진으로 만들어진 데이터셋이며, 비영리 목적 하의 사용, 배포 등이 가능한 데이터셋입니다. 단, 원작자에 대한 credit을 남겨야 하며, 이미지에 가해진 변형을 명시해야 합니다. U.S. Government Works 라이센스를 갖는 사진들은 제외되고, 그 외 (Creative Commons BY 2.0, Creative Commons BY-NC 2.0, Public Domain Mark 1.0, Public Domain CC0 1.0) 라이센스의 사진만 사용하였습니다.
  * **deepface** : 100k Faces 데이터셋과 FFHQ 데이터셋의 정답 및 기타 정보(나이, 성별, 인종)를 레이블링 하기 위해 사용되었습니다.
  * **MaskTheFace** : 100k Faces 데이터셋에 다양한 종류의 마스크를 추가하기 위해 사용되었습니다. 원 프로젝트 구성을 통해 다양한 마스크를 사진에 추가하였으며, [프로젝트 코드 변형](https://github.com/shhommychon/WrongMaskTheFace)을 통해 턱스크 등의 잘못된 마스크 또한 추가하였습니다.
* 원 대회에서의 데이터셋은 아시아인 남녀로만 구성되어 있었지만, 해당 데이터셋은 100k Faces와 FFHQ 데이터셋의 다양한 인종 구성을 이어받고 있습니다.
* 원 대회에서의 데이터셋과 유사하게, 나이는 10대부터 60대 까지 다양하게 분포하고 있습니다. 단, 50대 이후의 데이터를 구하기가 어려워서 레이블의 age boundary를 변형하였습니다.

### 새로운 데이터셋 구성

```
images/
├── 📂 train/
│   └── 📂 {id}_{gender}_{race}_{age}
│       ├── 😀 incorrect_mask.jpg
│       ├── 😀 mask1.jpg
│       ├── 😀 mask2.jpg
│       ├── 😀 mask3.jpg
│       ├── 😀 mask4.jpg
│       ├── 😀 mask5.jpg
│       └── 😀 normal.jpg
└── 📂 eval/
    ├── 😉 {id}_{gender}_{race}_{age}_{mask_type}.jpg
    ├── 😉 {id}_{gender}_{race}_{age}_{mask_type}.jpg
    └── 😉 {id}_{gender}_{race}_{age}_{mask_type}.jpg
```

* 전체 사람 명 수 : 15,300명 (train : 2,700명 / eval : 12,600명)

  * 원 대회의 데이터셋은 4,500명의 사진으로 구성되어 있었으며, 이 중 60%를 학습 데이터셋으로 활용하였습니다.
  * 100k Faces, FFHQ 데이터셋에서 Face Orientation, 나이, 성별 등 요소를 고려하여 subsampling 되었습니다.
  * Face Orientation은 [Irfan A. Khalid](https://www.linkedin.com/in/alghaniirfan/)의 [towardsdatascience 포스팅](https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600)의 Python 스크립트를 활용하여 파악하였습니다. 정면을 바라보는 사진들을 위주로 subsampling 하는 것을 목표로 하였으나, 부족한 나이/성별/인종의 경우에는 Face Orientation을 고려하지 않았습니다.

* 이미지 크기 : (384, 512)

  * 원 대회의 데이터셋 사진 크기와 동일합니다.
  * 100k Faces와 FFHQ 데이터셋 원본 이미지는 (1024, 1024) 입니다. 이를 (384, 512) 크기로 맞추기 위해서 위와 양 옆에 padding 및 resize가 진행되었습니다. 모든 비율은 확률 하에 결정되었습니다.

* train 데이터셋의 한 사람당 사진의 개수 : 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]

  * 원 대회의 train 데이터셋 구성과 동일합니다.
  * 100k Faces 데이터셋에서 가져온 사진(GAN으로 생성된 얼굴 사진)의 비율이 조금 더 높습니다.
  * 각 7개의 사진은 100k Faces 데이터셋 또는 FFHQ 데이터셋 중 임의의 사진 1장을 이용하여 제작되었습니다.
  * 특정 인종에 치우쳐져 있을 수 있습니다.

* eval 데이터셋의 한 사람당 사진의 개수 : 1 [마스크 착용, 이상하게 착용, 미착용 中 1장]

  * 원 대회의 eval 데이터셋 사진 수와 동일합니다.
  * FFHQ 데이터셋에서 가져온 사진(실제 사람을 촬영한 사진)의 비율이 훨씬 더 높습니다.
  * 각 사진은 100k Faces 데이터셋 또는 FFHQ 데이터셋 중 임의의 사진 1장을 이용하여 제작되었습니다.
  * train 데이터셋 보다 더 다양한 인종과 나이대를 커버하는 방향으로 제작하였습니다.
  * train 데이터셋에서 사용된 마스크의 종류보다 훨씬 더 다양한 종류의 마스크를 사용하여 제작하였습니다.

* Class Description

  * 마스크 착용여부, 성별, 나이를 기준으로 총 18개의 클래스가 있습니다.

  * | Class | Mask      | Gender | Age            |
    | ----- | --------- | ------ | -------------- |
    | 0     | Wear      | Male   | < 25           |
    | 1     | Wear      | Male   | >= 25 and < 40 |
    | 2     | Wear      | Male   | >= 40          |
    | 3     | Wear      | Female | < 25           |
    | 4     | Wear      | Female | >= 25 and < 40 |
    | 5     | Wear      | Female | >= 40          |
    | 6     | Incorrect | Male   | < 25           |
    | 7     | Incorrect | Male   | >= 25 and < 40 |
    | 8     | Incorrect | Male   | >= 40          |
    | 9     | Incorrect | Female | < 25           |
    | 10    | Incorrect | Female | >= 25 and < 40 |
    | 11    | Incorrect | Female | >= 40          |
    | 12    | Not Wear  | Male   | < 25           |
    | 13    | Not Wear  | Male   | >= 25 and < 40 |
    | 14    | Not Wear  | Male   | >= 40          |
    | 15    | Not Wear  | Female | < 25           |
    | 16    | Not Wear  | Female | >= 25 and < 40 |
    | 17    | Not Wear  | Female | >= 40          |

* ※ 유의할 점! ※

  * 본 데이터셋의 모든 나이, 성별 등 정보는 위에 명시된 Python 라이브러리들을 이용하여 레이블링 되었기 때문에 매우 부정확할 수 있습니다. 실질적인 모델의 학습보다, 컴퍼티션 후 후속 작업 및 코드 작동 여부 확인 등을 목적으로 사용하시는 것을 추천합니다.