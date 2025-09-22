# torchvision.transforms
- 이미지 전처리와 데이터 증강을 위한 도구
- 이 모듈은 이미지 데이터를 이용한 딥러닝 모델의 학습 효율을 높이고 데이터 준비 과정을 단순화 하는데 사용됨.

<br>

## 1. torchvision.transforms의 주요 역할
- 이미지 전처리: 크기 조정, 자르기, 회전.
- Data Augmentation (데이터 증강)
- 전처리 파이프라인 구축: 여러 변환을 조합하여 데이터 전처리 루틴을 생성.

    #### 지원하는 입력 형태
    - Pytorch의 Tensor 객체
    - PIL의 image 객체
    - NumPy의 ndarray 객체
<br>

    성능을 위해선, Pytorch의 Tensor 객체를 사용하는것이 권장됨.

<br>

## 2. 주요 함수와 사용법

### 2-1 이미지 크기 조정
1. transforms.Resize
    - 이미지를 지정된 크기로 변경.
    - transforms.v2.Resize로 대체하여 사용하는 것이 권장됨.
```python
#목표 크기 (height, width) 를 지정: 가로 세로비가 변경될 수 있음.
transform = transforms.Resize((128,128))

# 호출 가능한 객체이므로 다음과 같이 함수처럼 호출하여 사용.
resized_image = transform(image)
```

2. transforms.CenterCrop:
    - 중앙을 기준으로 이미지를 자름
```python
#이미지 중앙에서 지정된 크기 (height, width)만큼 잘라내기:
# 가로 세로 비가 유지됨.
# crop 크기가 원본보다 클 경우 패딩 추가
transform = transform.CenterCrop((100,100))

# 호출 가능한 객체이므로 다음과 같이 함수처럼 호출하여 사용.
cropped_image = transform(image)
```

3. transforms.RandomCrop
    - 임의의 위치에서 이미지를 자름.
    - padding으로 crop 전에 padding을 수행 가능.
```python
# 이미지에서 랜덤한 위치에서 지정된 크기 (height, width) 만큼 잘라내기: 
# Data Augmentation 으로 사용됨
# padding = 4: 원본 이미지 상하좌우에 4픽셀씩 패딩 추가 후 크롭
# 매번 호출할 때마다 다른 랜덤 위치에서 crop 수행
transform = transforms.RandoCrop((100, 100), padding=4)

# 호출 가능한 객체이므로 다음과 같이 함수처럼 호출하여 사용.
random_cropped_image = transform(image)
```
<br>

### 2-2 텐서 변환 (Conversion)

1. transforms.ToTensor
    - 이미지를 PyTorch 텐서로 변환.
    - 픽셀 값을 [0.0, 1.0]로 정규화(normalization):
        - 입력 데이터가 Pillow의 Image 객체인 경우 수행됨.
        - NumPy의 ndarray이면서 float 형이 아닐 경우 수행됨.
    
    - 입력데이터가 Pillow의 Image 객체인 경우,
        - 입력이 Width, Height, Channel 이라고 가정하고
        - 출력은 Channel, Width, Height로 변경됨.

```python
transform = transforms.ToTensor()
tensor_image = transform(image)
```

2. transforms.Normalize
    - 텐서를 normalization 하기 위해 표준화(standardization)를 진행.
    - 각 채널에 대해 (x-mean) / std 를 적용.
```python
transform = transform.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
normalized_image = transform(tensor_image)
```

### 2-3 Data Augmentation (데이터 증강)

1. transforms.RandomHorizontalFlip
    - 이미지 좌우 반전(Horizontal Flip)을 확률적으로 적용.
    - v2.RandomHorizontalFlip으로 대체를 권함

```py
# 이미지를 수평(좌우)으로 확률적으로 뒤집기
# p = 0.5: 50% 확률로 좌우 반전 수행
# 참고:
# p = 0.0: 절대 뒤집지 않음 (항상 원본)
# p = 1.0: 항상 뒤집음 (deterministic)
# p = 0.5: 절반 확률로 뒤집음 (일반적인 설정)
transform = tranforms.RandomHorizontalFlip(p=0.5)

# 호출 가능한 객체이므로 다음과 같이 함수처럼 사용.
flipped_image = transform(image)

#권장 방식
from torchvision.transforms import v2

transform = v2.RandomHorizontalFlip(p=0.5)
flipped_image = transform(image)
```

2. transforms.RandomRotation
    - 이미지를 랜덤 각도로 회전
        - 회전 시 빈 공간은 기본적으로 검은색(0)으로 채움.
        - 이미지 크기는 유지되므로 모서리 부분이 잘릴 수 있음

```py
# 이미지를 랜덤한 각도로 회전: 데이터 증강용으로 기하학적 변형 제공.
# degrees= 30: -30도에서 +30도 사이의 랜덤한 각도로 회전
# degrees=(15, 45): 15~45도 범위 지정 가능
# degrees=(-90, 90): 음수/양수로 시계/반시계 방향 제어
transform = transforms.RandomRotation(degrees=30)

# 다음과 같이 호출
rotated_image = transform(image)
```

3. transforms.RandomRotation
    - 이미지를 랜덤 각도로 회전.
        - 회전 시 빈 공간은 기본적으로 검은색(0)으로 채움
        - 이미지 크기는 유지되므로 모서리 부분이 잘릴 수 있음

```py
# 이미지를 랜덤한 각도로 회전: 데이터 증강용으로 기하학적 변형 제공.
# - degrees=30: -30도에서 +30도 사이의 랜덤한 각도로 회전
# - degrees=(15, 45): 최소 15도, 최대 45도 범위 지정 가능
# - degrees=(-90, 90): 음수/양수로 시계/반시계 방향 제어
transform = transforms.RandomRotation(degrees=30)

# callable객체이므로 다음과 같이 함수처럼 호출하여 사용.
rotated_image = transform(image)
```

  -  다음은 v2.RandomRotation의 사용예임.


```py
from torchvision.transforms import v2
import torch

# 기본 사용법
transform = v2.RandomRotation(
    degrees=30,                    # ±30도 범위
    interpolation=v2.InterpolationMode.BILINEAR,  # 보간법
    expand=False,                  # 이미지 크기 확장 여부
    center=None,                   # 회전 중심점
    fill=0                         # 빈 공간 채울 값/색상
)
```
<br>

### 2-4 기타

1. transforms.Grayscale
    - 이미지를 Gray Scale (L)로 변환
    - v2.Lambda로 대체하는 것을 권장함

```py
transform = transforms.Grayscale(num_output_channels=1)
gray_image = transform(image)
```

2. transforms.Lambda
    - 사용자 정의 변환을 구현한 lambda 표현을 적용.
    - 익명함수와 같이 사용시 다음의 이슈가 있음:
        - Multiprocessing의 경우 문제가 발생: DataLoader에서 num_workers를 1개 이상 사용 불가. 
        - 이는 pickle로 직렬화가 안되기 때문임.
        - 익명함수는 prototyping 등의 테스트에만 사용할 것.
    - 익명함수: 이름 없는 일회용 함수. 파이썬에서는 lambda로 만든다.
        - 특징: 한 줄로 된 간단한 로직만 구현 가능하다.
        - 주요 용도: map(), filter(), sorted()등 다른 함수의 인자로 함수 기능을 잠시 전달해야 할 때 코드를 간결하게 만들기 위해 사용한다.
        - 주의점: 복잡한 로직을 담기 시작하면 오히려 가독성이 떨어지므로, 그럴 땐 일반 함수(def)를 사용하는 것이 좋다.
    ```py
    # lambda 매개변수: 리턴할 표현식
    add_lambda = lambda a, b: a + b

    result = add_lambda(3,5) # 변수에 담아서 호출
    print(result)
    ```

<br>


## 3. 변환 파이프라인 구축
- torchvision.transforms.Compose를 통해 여러 transform 객체들을 연결하여 pipeline(파이프 라인)을 만들 수 있음.

```py
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

transformed_image = transform(image)
```

<br>

## 4. Custom Transform 작성법

### 4-1. 클래스 기반 커스텀 변환
- \_\_call\_\_ 메서드를 가진 클래스를 작성하여 새로운 transform을 정의 가능 (과거 방식).
```py
class CustomTransform:
    def __call__(self, img):
        return img.point(lambda x: x * 1.5) # 밝기를 1.5배 증가
```

- 현재는 torch.nn.Module을 상속하여 custom transform 클래스를 정의하는 것이 권해짐.
    - \_\_call\_\_ 함수에서 get_params 메서드를 호출하고, 이후 forward 메서드가 호출되는 순서로 동작함.
    - 때문에 get_params 메서드에서 변환에 필요한 파라미터들을 반환하고, 이들을 arguments로 이용하여 forward에서 변환이 되도록 수행함.

- get_params 패턴 도입에 따른 장점:
    - batch 처리 일관성: batch 전체에 동일한 파라미터 적용 가능함
        - v2에서 제공하는 transform들은 batch 차원을 가진 tensor 객체를 네이티브로 지원함.
    - 조건부 변환: 이미지 속성에 따른 적응적 파라미터 생성가능함.
    - 파라미터 의존성: 복합 변환에서 상호 연관된 파라미터 관리 가능.
    - 디버깅 지원: 파라미터 로깅 및 재현 가능한 변환
    - 성능 최적화: 파라미터 계산과 변환 로직 분리로 효율성 향상

- 다음 코드는 get_params를 도입한 CustomTransform 클래스를 만드는 간단한 예이다:

```py
import torch
from torchvision.transforms import v2
import random

class AdvancedCustomTransform(torch.nn.Module):
    """v2 transforms의 표준 패턴을 따르는 커스텀 변환"""
    
    def __init__(self, brightness_range=(0.8, 1.2), rotation_range=(-30, 30)):
        super().__init__()
        self.brightness_range = brightness_range
        self.rotation_range = rotation_range
    
    def get_params(self):
        """
        변환에 필요한 랜덤 파라미터들을 미리 계산
        forward() 호출 전에 __call__에서 자동 호출됨
        """
        brightness_factor = random.uniform(*self.brightness_range)
        rotation_angle = random.uniform(*self.rotation_range)
        apply_blur = random.random() > 0.5
        
        return {
            'brightness_factor': brightness_factor,
            'rotation_angle': rotation_angle, 
            'apply_blur': apply_blur
        }
    
    def forward(self, img, **params):
        """
        실제 변환 수행 - get_params에서 생성된 파라미터 사용
        동일한 파라미터로 배치의 모든 이미지에 동일한 변환 적용 가능
        """
        # 1. 밝기 조정
        img = v2.functional.adjust_brightness(img, params['brightness_factor'])
        
        # 2. 회전
        img = v2.functional.rotate(img, angle=params['rotation_angle'])
        
        # 3. 조건부 블러
        if params['apply_blur']:
            img = v2.functional.gaussian_blur(img, kernel_size=5, sigma=1.0)
        
        return img
    
    def __call__(self, img):
        """
        표준 호출 인터페이스
        1. get_params() 호출하여 랜덤 파라미터 생성
        2. forward(img, **params) 호출하여 변환 수행
        """
        params = self.get_params()
        return self.forward(img, **params)

# 사용법
advanced_transform = AdvancedCustomTransform()
```
### 4-2 함수 기반 커스텀 변환
- 단순한 변환은 함수로 구현할 수 있음.
    - transforms.Lamdba를 활용하여 간단히 추가 가능.
    - 또는 functional 모듈의 함수들을 이용할 수도 있음.

```py
def custom_transform(img):
    """PIL 이미지를 흑백으로 변환하는 함수"""
    return img.convert("L")  # PIL의 convert("L") - RGB를 Grayscale로 변환

transform = transforms.Compose([
    transforms.Resize((128, 128)),        # 이미지 크기를 128x128로 조정
    transforms.Lambda(custom_transform),  # 커스텀 함수를 transform으로 래핑 
                                          # 문제: pickle 불가, multiprocessing 호환 안됨
    transforms.ToTensor()                 # PIL → Tensor 변환 + [0,255] → [0,1] 정규화
])
```

- 역시 v2로 대체하는 것이 권장됨:
```py
from torchvision.transforms import v2

# v2 권장 방식: Global 함수 + Lambda (최후 수단)
def safe_grayscale(img):
    """Global scope의 함수 - pickle 가능"""
    return v2.functional.rgb_to_grayscale(img, num_output_channels=1)

transform_v2_lambda = v2.Compose([
    v2.Resize((128, 128)),                    # 크기 조정
    v2.Lambda(safe_grayscale),                # 글로벌 함수는 pickle 가능 
                                              # (하지만 클래스 방식 권장)
    v2.ToDtype(torch.float32, scale=True)     # Tensor 변환 + 정규화
])
```

- 가급적이면 CustomTransform 클래스로 만들어서 처리하는 것이 권장됨.

```py
from torchvision.transforms import v2

# v2 권장 방식: 커스텀 클래스 (고급 사용자용)
class GrayscaleTransform(torch.nn.Module):
    """RGB를 흑백으로 변환하는 안전한 커스텀 변환"""
    
    def forward(self, img):
        # v2.functional 사용으로 PIL/Tensor 모두 처리
        return v2.functional.rgb_to_grayscale(img, num_output_channels=1)

transform_v2_custom = v2.Compose([
    v2.Resize((128, 128)),                    # 크기 조정
    GrayscaleTransform(),                     # 커스텀 흑백 변환 (pickle 가능, multiprocessing 호환)
    v2.ToDtype(torch.float32, scale=True)     # Tensor 변환 + 정규화
])
```
## 5. DataLoader와 결합하여 GPU로 데이터 옮기기

- torchvision.transforms는
    - Dataset과 결합하여 대규모 데이터셋에 자동으로 변환을 적용.
    - DataLoader는 Dataset 객체를 통해 데이터를 로드할 때, CPU에서 동작함.
    - 때문에 Dataset 객체의 \_\_getitem\_\_ 메서드 내부에서 호출되는 Transform 객체도 CPU에서 동작.
    - 때문에, DataLoader가 Transformdl적용된 batch를 반환하면 이를 GPU로 이동시켜야 한다.
- 때문에, 학습 중 GPU를 효율적으로 활용하려면 최종 데이터(Transform이 적용된 Tensor 객체)를 GPU로 이동시켜야함. 

1. 데이터셋 및 DataLoader 정의
```py
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
dataset = ImageFolder(
            root='path_to_images', 
            transform=transform,
          )
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
```

2. GPU로 데이터 전송 및 학습 루프
```py
import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 모델 정의
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 128 * 128, 10)
).to('cuda')  # 모델을 GPU로 이동

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(5):
    for images, labels in dataloader:
        images = images.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)

        # 순전파 및 역전파
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')
```
## 6. 팁
- Data Augmentation과 Pre-Processing 분리:
    - Data Augmentation은 Training 단계에서만 사용
    - Evaluation/Test 단계에서는 Pre-Processing만 적용.
    - 가장 명확한 방법은 각 단계별로 개별 Dataset 객체를 사용하는 것임.
    - 개별 Dataset은 각기 다른 transform을 설정

- 고정 메모리와 비동기 전송 활용:
    - pin_memory=True와
    - non_blocking=True를 사용해
    - GPU 전송 성능을 최적화.

- GPU 활용 여부 확인:
    - 학습 전에 GPU가 사용 가능한지 확인.
    ```py
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ```
 