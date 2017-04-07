import subprocess as sp
import os
import time 
from watchdog.observers import Observer  
from watchdog.events import PatternMatchingEventHandler 

spearmint_iters = 5;
K = 0

class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.out"]
    #start running the program

    def process(self, event):
    	count = __checkFileCount__(1)
    	if count >= spearmint_iters:
    		#stop the process and stop listening
    		print "\r\r\r\r\r\r\r\r\##################################################################\tAttempting to stop observer\t\##################################################################\r\r\r\r\r\r\r\r"
    		global check
    		check = 1



    def on_created(self, event):
        self.process(event)


def __checkFileCount__(k):
	path, dirs, files = os.walk("/home/tomj/Documents/4YP/densecrf/CV/AllParameters-QPNC/CV" + str(K) + "/output").next()
	#path, dirs, files = os.walk("/home/tomj/Documents/4YP/densecrf/CV/test/"  + str(K)).next()
	file_count = len(files)
	return file_count



if __name__ == '__main__':
	global check
	for k in range(1,5):
		print "Starting the " + str(k) + "th observer."
		K = k #terrible way of 'passing' an argument to the class
		check = 0
		extProc = sp.Popen(['python','test.py', str(K)]) # runs myPyScript.py 
		observer = Observer()
		observer.schedule(MyHandler(), "/home/tomj/Documents/4YP/densecrf/CV/AllParameters-QPNC/CV" + str(K) + "/output")
		observer.start()
		while (check == 0):
			time.sleep(1)
		print "Stopping observer"
		observer.stop()
		observer.join()
		print "Stopping process"
		extProc.terminate()

