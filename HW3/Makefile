.PHONY: oscillator, poisson, clean

Part := 1,2,3,4,5

all: oscillator, poisson

default: all

oscillator:
	python3 oscillator.py --part=${Part}

poisson:
	python3 poisson.py --part=${Part}

clean:
	rm *.jpeg
