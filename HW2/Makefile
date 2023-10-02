.PHONY: carbon golf clean

WIDTH := 10,100,1000
THETA := 45,30,15,9

all: carbon golf

default: all


carbon:
	python3 carbon.py --plot=${WIDTH}


golf:
	python3 golf.py --plot=${THETA}


clean:
	rm -f *.jpeg
