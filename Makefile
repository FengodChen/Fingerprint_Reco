deps:
	pip3 install -r requirements.txt --user
	apt install python3-opencv -y

run:
	python3 main.py