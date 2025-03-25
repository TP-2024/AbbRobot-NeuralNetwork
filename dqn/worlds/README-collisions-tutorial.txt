.proto file Irb4600-40 ramena skopirovany z jeho oficialneho githubu. Donho pridane TouchSensory s unikatnymi nazvami. Skopirovane dependencie ramena do protos adresara a prelinkovane v .proto a .wbt subore. Controller na toto este nie je nastaveny. Je velmi dolezite v controlleri pred main loopom (po inicializacii motorov a sensorov) nechat robota spravit jeden (pripadne viac) stepov (popisane v controller priklade).


Controller:

Accessnutie TouchSensorov rovnako ako motorov a klasickych sensorov (A - F touch sensor)

robot.getDevice("C touch sensor")

Citanie hodnot TouchSensorov:

if [TouchSensor].getValue() == 1.0:
	nastala kolizia



PRIKLAD CONTROLLERU:

# ziskanie vsetkych touch sensorov
touchs = []
touchs.append(robot.getDevice("A touch sensor"))  # rameno
touchs.append(robot.getDevice("B touch sensor"))  # bicak
touchs.append(robot.getDevice("C touch sensor"))  # laket
touchs.append(robot.getDevice("D touch sensor"))  # predlaktie
touchs.append(robot.getDevice("E touch sensor"))  # dlan
touchs.append(robot.getDevice("F touch sensor"))  # prsty

# aktivovanie touch sensorov
for touch in touchs:
    touch.enable(timestep)

# !!! VELMI DOLEZITE !!!
# webots pri restarte dobre neresetne sensor states
# cize sa po rune s koliziou a nasledovnom restarte
# ihned aktivuje falosna kolizia
robot.step(timestep)

# main loop:
while robot.step(timestep) != -1:
	for touch in touchs:
		if touch.getValue() == 1.0:
			print("COLISION DETECTED!!")