# OS

<br>

## 파이썬 OS 라이브러리
- OS 라이브러리는 운영 체제와 상호작용하는 다양한 기능 제공
- 파일 및 디렉토리 관리, 시스템 명령어 실행, 경로 관련 작업 등의 기능

<br>

----

<br>

## OS 모듈 함수

1. os.name : 운영 체제의 이름
   - os.name 함수는 현재 실행되고 있는 운영 체제의 이름을 반환
  
```
  print(os.name)
```

2. os.getcwd() : 현재 작업 디렉토리 불러오기
   - 현재 작업 디렉토리를 반환하는 함수이다. 이는 프로그램이 현재 어디에 위치하는지 알 수 있다.
```
current_directory = os.getcwd()
print(current_directory)
```
