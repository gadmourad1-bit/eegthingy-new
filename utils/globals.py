from utils.devices import OpenBCI, Vernier, Noraxon, Joystick

"""This file is used to create global instances of the different bridges, so they can be accessed from anywhere in the application."""

openbci = OpenBCI()
openbci.open()

vernier = Vernier()
vernier.open()

noraxon = Noraxon()
#noraxon.open("192.168.137.148", 9220) # This is commented out because the Noraxon bridge is not currently being used.

joystick = Joystick()
joystick.open()