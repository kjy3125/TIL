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
3. os.makedirs(): 디렉토리 생성하기
    - 새 디렉토리를 생성할 때 사용
```
os.makedirs('new_directory')
```

4. os.rmdir(): 디렉토리 삭제
    - 빈 디렉토리를 삭제할 때 사용
```
os.rmdir('directory_to_remove') 
```
5. os.remove(): 파일 삭제하기
    - 특정 파일을 삭제할 때 사용
```
os.remove('file_to_remove.txt')
```
6. os.rename(): 파일 및 디렉토리 이름 변경
    - 파일 또는 디렉토리의 이름을 변경할 때 사용
```
os.rename('old_name.txt', 'new_name.txt')
```
7. os.stat(): 파일의 속성 가져오기
    - 파일의 다양한 속성(예: 크기, 수정 시간 등) 반환
```
file_info = os.stat('example.txt')
print(file_into.st_size)
```
8. os.listdir(): 디렉토리 내의 파일 및 서브 디렉토리 리스트
 - 현재 디렉토리 또는 지정된 디렉토리 내의 파일과 서브 디렉토리 목록 반환
```
directory_content = os.listdir('.')
print(directory_content)
```
9. os.system(): 시스템 명령어 실행하기
    - os의 시스템 명령어를 직접 실행
```
os.system('echo Hello World')
```
10. os.environ: 환경 변수 정보
    - 현재 환경 변수의 딕셔너리 제공
```
print(os.environ['PATH'])
```
11. os.getenv() & os.putenv() : 환경 변수값 가져오기 및 설정하기
    - 환경 변수의 값을 가져오거나 설정할 때 사용
```
# 값을 가져옴 
path_value = os.getenv('PATH')
# 값을 설정
os.putenv('MY_VARIABLE', 'value')
```
12. os.path.join(): 경로 결합하기
    - 두 개 이상의 경로 구성 요소를 안전하게 결합 가능
```
full_path = os.path.join('/home', 'user', 'documents', 'file.txt')
print(full_path)
```
13. os.path.split() 및 os.path.exists(): 경로 분리 및 경로의 존재 확인하기
    - 경로를 디렉토리와 파일로 분리하거나, 경로의 존재 확인
```
directory, filename = os.path.split(full_path)
print(directory, filename)

#경로의 존재 확인
print(os.path.exists('/home/user/documents'))
```
14. os.path.isdir() 및 os.path.isfile(): 디렉토리 및 파일 여부 확인
    - 특정 경로가 디렉토리인지, 파일 인지를 확인하는 함수
```
print(os.path.isdir('/home/user'))
print(os.path.isfile(full_path))
```
15. os.open(),os.read(),os.write(),os.close(): 기본적인 파일 작업
    -이들 함수를 통해 파일을 열고, 읽고, 쓰고, 닫는 기본 작업 수행

```
fd = os.open("example.txt", os.O_RDWR)
content = os.read(fd,15)
os.write(fd, b'Hello, World!')
os.close(fd)
```

16. os.pipe(): 파이프 사용하기 
    - 파이프를 생성하여 프로세스 간의 통신을 가능하게 함
    ```
    for dirpath, dirnames, filenames in os.walk('.'):
    print(f"Current directory: {dirpath}")
    for file in filenames:
        print(file)
    ```

17. os.spwan() 및 os.kill(): 자식 프로세스 시작 및 종료
    - os.spwan() 함수는 새로운 프로세스 시작, os.kill()은 프로세스를 종료
    ```
    pid = os.spwan(os.P_DETACH, 'path/to/program', 'arg1', 'arg2')
    os.kill(pid, signal.SIGTERM)
    ```

<br>

----

<br>

## 주의사항 및 팁
- os 라이브로리를 사용할 때는 특히 파일 삭제 또는 시스템 명령어 실행과 같은 민감한 작업을 수행할 때 주의해야 한다. 항상 코드를 실행하기 전에 테스트를 수행하고, 가능한 한 예외 처리를 포함시키는 것이 좋다.
