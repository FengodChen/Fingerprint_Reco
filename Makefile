deps:
	pip3 install -r requirements.txt
	apt install python3-opencv -y

train:
	python3 train.py

test:
	python3 test.py