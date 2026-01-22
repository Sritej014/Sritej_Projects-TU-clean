// ***** 0. Documentation Section *****
// SwitchLEDInterface.c for Lab 8
// Runs on LM4F120/TM4C123
// Use simple programming structures in C to toggle an LED
// while a button is pressed and turn the LED on when the
// button is released.  This lab requires external hardware
// to be wired to the LaunchPad using the prototyping board.
// January 15, 2016
//      Jon Valvano and Ramesh Yerraballi

// ***** 1. Pre-processor Directives Section *****
#include "TExaS.h"
#include "tm4c123gh6pm.h"

// ***** 2. Global Declarations Section *****
volatile unsigned long SW1; 
// FUNCTION PROTOTYPES: Each subroutine defined
void DisableInterrupts(void); // Disable interrupts
void EnableInterrupts(void);  // Enable interrupts
void Delay100ms(unsigned long msec);

// ***** 3. Subroutines Section *****

// PE0, PB0, or PA2 connected to positive logic momentary switch using 10k ohm pull down resistor
// PE1, PB1, or PA3 connected to positive logic LED through 470 ohm current limiting resistor
// To avoid damaging your hardware, ensure that your circuits match the schematic
// shown in Lab8_artist.sch (PCB Artist schematic file) or 
// Lab8_artist.pdf (compatible with many various readers like Adobe Acrobat).
int main(void){ 
	volatile unsigned long delay; 
//**********************************************************************
// The following version tests input on PE0 and output on PE1
//**********************************************************************
  TExaS_Init(SW_PIN_PE0, LED_PIN_PE1, ScopeOn);  // activate grader and set system clock to 80 MHz
	SYSCTL_RCGC2_R |= 0x10;
	delay = SYSCTL_RCGC2_R;
	GPIO_PORTE_DIR_R |= 0x02;//out PE1
	
	GPIO_PORTE_DIR_R &= ~0x01; //in PE0
	GPIO_PORTE_AFSEL_R &= ~0x03; // no alterante for PE1, PE0 clear the bits
	GPIO_PORTE_AMSEL_R &= ~0x03; // no analog for pE1, pE0 clear the bits
	GPIO_PORTE_PCTL_R &= ~0x000000FF; // clear for PE1 and PE0 16 bit field!
	GPIO_PORTE_DEN_R |= 0x03;//set PE1, PE0 for digital pins
	GPIO_PORTE_DATA_R |= 0x02; //SET PE1 as output is 1 at beginning 
	
  
	
  EnableInterrupts();           // enable interrupts for the grader
  while(1){
    SW1= GPIO_PORTE_DATA_R & 0x01; //read pE0 as input to PE1
		GPIO_PORTE_DATA_R |= 0x02;
		Delay100ms(1);

		if(SW1 == 1)
			GPIO_PORTE_DATA_R ^= 0x02;
		
  }
  
}

void Delay100ms(unsigned long msec)
{
	unsigned long i; 
	while(msec>0)
	{
		i = 1600;
			while(i>0)
				i--;
		msec--;
	}	
		
}