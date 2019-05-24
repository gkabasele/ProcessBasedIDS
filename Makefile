RUNARGS= --conf $(CONFIG) --infile $(TRACES)
init:
	pip3 install -r requirements.txt

test:
	py.tests tests

run:
	python3 ./ids/main.py $(RUNARGS)

.PHONY: init test run
