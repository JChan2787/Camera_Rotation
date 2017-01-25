Camera Control README

Submitted By		: Jose Vicente D. Chan Jr.
Campus Wide ID		: 892984154
Class			: CPSC 484
Time			: Tue & Thu @ 1:00PM

Assignment 5 -- Camera Control

Summary:

For this assignment, we are to implement a camera control scheme
	using functions we have implemented in our GFXMath.h.

	cameraRotateLeft & cameraRotateUp are the two main actors
	for this assignment. These two functions will be utilizing
	the Rodrigues rotation formula for its rotation matrices. 
	
	To run the program, a makefile is already provided so all
	is needed is to type in the make command in the terminal. 
	
	Camera Control Scheme:

	'-' & '+'     : decrease and increase the angle in degrees
	'r'	      : re-initialize the camera to its default position
	'LEFT-ARROW'  : rotate the camera towards the right
	'RIGHT-ARROW' : rotate the camera towards the left
	'UP-ARROW'    : rotate the camera downwards
	'DOWN-ARROW'  : rotate the camera upwards
	'q' & 'esc'   : terminate the program.

	The program is running smoothly upon running. Controls are 
	working perfectly, with no glitches or errors occuring thus
	far. Camera will rotate around the teapot based on the user
	input, upwards, downwards, left, or right. 
		

