import RPi.GPIO as GPIO

from RPLCD.gpio import CharLCD

#CharLCD is a Python library for accessing Adafruit character LCDs from a Raspberry Pi


lcd = CharLCD(pin_rs=19, pin_e=16, pin_rw=None, 
		pins_data=[23, 17, 21, 22],
		numbering_mode=GPIO.BOARD,
		cols=16, rows=2, dotsize=8)

#Include libraries
import RPi.GPIO as GPIO 
import time
from RPLCD.gpio import CharLCD

# Configure the LCD
lcd = (pin_rs = 19, pin_rw = None, pin_e = 16, pins_data = [21,18,23,24], 
numbering_mode = GPIO.BOARD)

# Create a variable ‘number’ 
number = 0

# Main loop
while(True):
# Increment the number and then print it to the LCD number = number + 1
lcd.clear()
lcd.write_string(“Count: “+ str(number))
lcd.write_string(“Hello World”)
lcd.cursor_pos = (2, 0)
time.sleep(1) 

lcd.close() 
GPIO.cleanup()

