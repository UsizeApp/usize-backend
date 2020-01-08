
DBG = 0

def printd(*args, **kwargs):
	global DBG
	if DBG:
		print(*args, **kwargs)

	return

def enablePrint(d):
	global DBG
	DBG = d
