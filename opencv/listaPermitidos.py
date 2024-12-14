import time
import pigpio

import RP1.GPIO as GPIO
import time
GPIO.setwarnings(false)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18.GPIO,OUT)

from time import sleep
import pygame.mixer
import sys
pygame.mixer.init(44100,-16.2,4096)

class flabianos:
    
    def __init__(self):
        self.Invitados=['ojos abiertos']
        
        def TuSuTuNo(self f, EllosSi):
            if EllosSi in self.Invitados:
                print('Despierto')
            else:
                print('ojos cerrados alerta')
                segundos_d=3
                inicio=time.time()
                final=time.time()+segundos_d
                GPIO_output(18,True)
                while inicio<=final:
                    inicio = time.time()
                else:
                    GPIO.output(18,False)
                print ('entro aqui')