import numpy as np 
import csv

def saveOnePlayerPoint(player_id, point, fr):
	filename = "player_{id}_points.csv".format(id = player_id)
	with open(filename, 'a') as csvfile:
		# fieldnames = ['id', 'point0', 'point1']
		writer = csv.writer(csvfile, delimiter=',',)
	
		# writer.writerow({'id': player_id , 'point0': point[0], 'point1': point[1]})
		writer.writerow([player_id, point[0], point[1], fr])
	
	return

def main():
	### test ### 
	saveOnePlayerPoint("B1", [12,12], 0)
	saveOnePlayerPoint("B1", [24,23], 1)
	return

if __name__ == "__main__":
	main()