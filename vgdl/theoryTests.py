from .theory_template import *
from .rlenvironmentnonstatic import defInputGame, createRLInputGame, createRLInputGameFromStrings
from .metaplanner import observe
from .ontology import *
if __name__ == "__main__":

	filename = "examples.gridphysics.simpleGame_preconditions"
	gameString, levelString = defInputGame(filename, randomize=True)
	rleCreateFunc = lambda: createRLInputGameFromStrings(gameString, levelString)
	rle = rleCreateFunc()
	allObjects= rle._game.getObjects()

	observe(rle, 5)
	spriteTypeHypothesis = sampleFromDistribution(rle._game.spriteDistribution, allObjects)
	gameObject = Game(spriteInductionResult=spriteTypeHypothesis)
	initialTheory = gameObject.buildGenericTheory(spriteTypeHypothesis)

	####### worked #########
	gameState1 = {'ended': False, 'score': 0, 'objects': {'wall': {(300, 0): {'y': 0, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430774864}, (90, 0): {'y': 0, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430472784}, (120, 150): {'y': 150, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430776400}, (30, 150): {'y': 150, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430776208}, (240, 150): {'y': 150, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430776656}, (0, 150): {'y': 150, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430776144}, (210, 150): {'y': 150, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430776592}, (270, 0): {'y': 0, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430774800}, (60, 0): {'y': 0, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430627024}, (270, 150): {'y': 150, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430776720}, (360, 120): {'y': 120, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776080}, (30, 0): {'y': 0, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430589584}, (90, 150): {'y': 150, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430776336}, (60, 150): {'y': 150, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430776272}, (360, 30): {'y': 30, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775184}, (240, 0): {'y': 0, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430774736}, (210, 0): {'y': 0, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430774672}, (0, 0): {'y': 0, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430588560}, (360, 150): {'y': 150, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776912}, (0, 120): {'y': 120, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775824}, (180, 0): {'y': 0, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430774608}, (360, 60): {'y': 60, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775312}, (0, 30): {'y': 30, 'x': 0, 'direction': None, 'speed': None, \
	'ID': 4430775056}, (150, 0): {'y': 0, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430774544}, (0, 90): {'y': 90, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775376}, (330, 150): {'y': 150, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776848}, (180, 150): {'y': 150, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430776528}, (150, 150): {'y': 150, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430776464}, (360, 0): {'y': 0, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430774992}, (300, 150): {'y': 150, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430776784}, (330, 0): {'y': 0, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430774928}, (120, 0): {'y': 0, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430774480}, (360, 90): {'y': 90, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775760}, (0, 60): {'y': 60, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775248}}, 'medicine': {}, 'avatar': {(60, 30): {'direction': None, 'y': 60, 'x': 60, 'speed': 1, 'ID': 4430775888, 'resources': {'medicine': 2}}}, 'poison': {(300, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 300, 'speed': None, 'ID': 4430775632}, (210, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 210, 'speed': None, 'ID': 4430775440}, (270, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 270, 'speed': None, 'ID': 4430775568}, (240, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 240, 'speed': None, 'ID': 4430775504}, (330, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 330, 'speed': None, 'ID': 4430775696}, (240, 120): {'direction': None, 'limit': 3, 'y': 120, 'x': 240, 'speed': None, 'ID': 4430775952}}, 'goal': {(330, 120): {'y': 120, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776016}}}, 'win': None}
	gameState2 = {'ended': False, 'score': 0, 'objects': {'wall': {(300, 0): {'y': 0, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430774864}, (90, 0): {'y': 0, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430472784}, (120, 150): {'y': 150, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430776400}, (30, 150): {'y': 150, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430776208}, (240, 150): {'y': 150, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430776656}, (0, 150): {'y': 150, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430776144}, (210, 150): {'y': 150, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430776592}, (270, 0): {'y': 0, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430774800}, (60, 0): {'y': 0, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430627024}, (270, 150): {'y': 150, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430776720}, (360, 120): {'y': 120, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776080}, (30, 0): {'y': 0, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430589584}, (90, 150): {'y': 150, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430776336}, (60, 150): {'y': 150, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430776272}, (360, 30): {'y': 30, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775184}, (240, 0): {'y': 0, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430774736}, (210, 0): {'y': 0, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430774672}, (0, 0): {'y': 0, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430588560}, (360, 150): {'y': 150, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776912}, (0, 120): {'y': 120, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775824}, (180, 0): {'y': 0, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430774608}, (360, 60): {'y': 60, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775312}, (0, 30): {'y': 30, 'x': 0, 'direction': None, 'speed': None, \
	'ID': 4430775056}, (150, 0): {'y': 0, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430774544}, (0, 90): {'y': 90, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775376}, (330, 150): {'y': 150, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776848}, (180, 150): {'y': 150, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430776528}, (150, 150): {'y': 150, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430776464}, (360, 0): {'y': 0, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430774992}, (300, 150): {'y': 150, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430776784}, (330, 0): {'y': 0, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430774928}, (120, 0): {'y': 0, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430774480}, (360, 90): {'y': 90, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775760}, (0, 60): {'y': 60, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775248}}, 'medicine': {}, 'avatar': {(60, 30): {'direction': None, 'y': 30, 'x': 60, 'speed': 1, 'ID': 4430775888, 'resources': {'medicine': 2}}}, 'poison': {(300, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 300, 'speed': None, 'ID': 4430775632}, (210, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 210, 'speed': None, 'ID': 4430775440}, (270, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 270, 'speed': None, 'ID': 4430775568}, (240, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 240, 'speed': None, 'ID': 4430775504}, (330, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 330, 'speed': None, 'ID': 4430775696}, (240, 120): {'direction': None, 'limit': 3, 'y': 120, 'x': 240, 'speed': None, 'ID': 4430775952}}, 'goal': {(330, 120): {'y': 120, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776016}}}, 'win': None}
	gameState3 = {'ended': False, 'score': 0, 'objects': {'wall': {(300, 0): {'y': 0, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430774864}, (90, 0): {'y': 0, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430472784}, (120, 150): {'y': 150, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430776400}, (30, 150): {'y': 150, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430776208}, (240, 150): {'y': 150, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430776656}, (0, 150): {'y': 150, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430776144}, (210, 150): {'y': 150, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430776592}, (270, 0): {'y': 0, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430774800}, (60, 0): {'y': 0, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430627024}, (270, 150): {'y': 150, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430776720}, (360, 120): {'y': 120, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776080}, (30, 0): {'y': 0, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430589584}, (90, 150): {'y': 150, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430776336}, (60, 150): {'y': 150, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430776272}, (360, 30): {'y': 30, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775184}, (240, 0): {'y': 0, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430774736}, (210, 0): {'y': 0, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430774672}, (0, 0): {'y': 0, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430588560}, (360, 150): {'y': 150, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776912}, (0, 120): {'y': 120, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775824}, (180, 0): {'y': 0, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430774608}, (360, 60): {'y': 60, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775312}, (0, 30): {'y': 30, 'x': 0, 'direction': None, 'speed': None, \
	'ID': 4430775056}, (150, 0): {'y': 0, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430774544}, (0, 90): {'y': 90, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775376}, (330, 150): {'y': 150, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776848}, (180, 150): {'y': 150, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430776528}, (150, 150): {'y': 150, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430776464}, (360, 0): {'y': 0, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430774992}, (300, 150): {'y': 150, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430776784}, (330, 0): {'y': 0, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430774928}, (120, 0): {'y': 0, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430774480}, (360, 90): {'y': 90, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775760}, (0, 60): {'y': 60, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775248}}, 'medicine': {}, 'avatar': {(210, 90): {'direction': None, 'y': 90, 'x': 180, 'speed': 1, 'ID': 4430775888, 'resources': {'medicine': 0}}}, 'poison': {(300, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 300, 'speed': None, 'ID': 4430775632}, (240, 120): {'direction': None, 'limit': 3, 'y': 120, 'x': 240, 'speed': None, 'ID': 4430775952}, (270, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 270, 'speed': None, 'ID': 4430775568}, (240, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 240, 'speed': None, 'ID': 4430775504}, (330, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 330, 'speed': None, 'ID': 4430775696}}, 'goal': {(330, 120): {'y': 120, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776016}}}, 'win': None}
	gameState4 = {'ended': False, 'score': 0, 'objects': {'wall': {(300, 0): {'y': 0, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430774864}, (90, 0): {'y': 0, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430472784}, (120, 150): {'y': 150, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430776400}, (30, 150): {'y': 150, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430776208}, (240, 150): {'y': 150, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430776656}, (0, 150): {'y': 150, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430776144}, (210, 150): {'y': 150, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430776592}, (270, 0): {'y': 0, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430774800}, (60, 0): {'y': 0, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430627024}, (270, 150): {'y': 150, 'x': 270, 'direction': None, 'speed': None, 'ID': 4430776720}, (360, 120): {'y': 120, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776080}, (30, 0): {'y': 0, 'x': 30, 'direction': None, 'speed': None, 'ID': 4430589584}, (90, 150): {'y': 150, 'x': 90, 'direction': None, 'speed': None, 'ID': 4430776336}, (60, 150): {'y': 150, 'x': 60, 'direction': None, 'speed': None, 'ID': 4430776272}, (360, 30): {'y': 30, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775184}, (240, 0): {'y': 0, 'x': 240, 'direction': None, 'speed': None, 'ID': 4430774736}, (210, 0): {'y': 0, 'x': 210, 'direction': None, 'speed': None, 'ID': 4430774672}, (0, 0): {'y': 0, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430588560}, (360, 150): {'y': 150, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430776912}, (0, 120): {'y': 120, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775824}, (180, 0): {'y': 0, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430774608}, (360, 60): {'y': 60, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775312}, (0, 30): {'y': 30, 'x': 0, 'direction': None, 'speed': None, \
	'ID': 4430775056}, (150, 0): {'y': 0, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430774544}, (0, 90): {'y': 90, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775376}, (330, 150): {'y': 150, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776848}, (180, 150): {'y': 150, 'x': 180, 'direction': None, 'speed': None, 'ID': 4430776528}, (150, 150): {'y': 150, 'x': 150, 'direction': None, 'speed': None, 'ID': 4430776464}, (360, 0): {'y': 0, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430774992}, (300, 150): {'y': 150, 'x': 300, 'direction': None, 'speed': None, 'ID': 4430776784}, (330, 0): {'y': 0, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430774928}, (120, 0): {'y': 0, 'x': 120, 'direction': None, 'speed': None, 'ID': 4430774480}, (360, 90): {'y': 90, 'x': 360, 'direction': None, 'speed': None, 'ID': 4430775760}, (0, 60): {'y': 60, 'x': 0, 'direction': None, 'speed': None, 'ID': 4430775248}}, 'medicine': {}, 'avatar': {}, 'poison': {(300, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 300, 'speed': None, 'ID': 4430775632}, (240, 120): {'direction': None, 'limit': 3, 'y': 120, 'x': 240, 'speed': None, 'ID': 4430775952}, (270, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 270, 'speed': None, 'ID': 4430775568}, (330, 90): {'direction': None, 'limit': 3, 'y': 90, 'x': 330, 'speed': None, 'ID': 4430775696}}, 'goal': {(330, 120): {'y': 120, 'x': 330, 'direction': None, 'speed': None, 'ID': 4430776016}}}, 'win': None}
	event1 = {'agentState': {'medicine':0}, 'agentAction': (1,0), \
	'effectList': [('changeResource', 'DARKBLUE', 'WHITE', 'medicine', 1), ('killSprite', 'WHITE', 'DARKBLUE')],\
	 'gameState': gameState1}
	event2 = {'agentState': {'medicine':1}, 'agentAction': (1,0), \
	'effectList': [('stepBack', 'DARKBLUE', 'BLACK')],\
	 'gameState': gameState2}
	event3 = {'agentState': {'medicine':1}, 'agentAction': (1,0), \
	'effectList': [('changeResource', 'DARKBLUE', 'BROWN', 'medicine', -1), ('killSprite', 'BROWN', 'DARKBLUE')],\
	 'gameState': gameState3}
	event4 = {'agentState': {'medicine':0}, 'agentAction': (1,0), \
	'effectList': [('changeResource', 'DARKBLUE', 'BROWN', 'medicine', -1), ('killSprite', 'BROWN', 'DARKBLUE'), \
	('killSprite', 'DARKBLUE', 'BROWN')],\
	 'gameState': gameState4}

	# Testing the two ways in which you could be led to need preconditions
	# eventList = [event1, event2, event3, event4]
	eventList = [event4, event2, event3, event1]


	terminationCondition = {'ended': False, 'win':False, 'time':5}
	trace = ([TimeStep(e['agentAction'], e['agentState'], e['effectList'], e['gameState']) for e in eventList], terminationCondition)
	

	hypotheses = list(gameObject.runInduction(gameObject.spriteInductionResult, trace, 20, verbose=False)) ##if you resample or run sprite induction, this 

	print("found", len(hypotheses), "hypotheses")

	embed()

symbolDict = generateSymbolDict(rle)

game, level, symbolDict, immovables = writeTheoryToTxt(rle, hypotheses[0], symbolDict, \
"./examples/gridphysics/theorytest.py", goalLoc=(3,3))

vrle = createMindEnv(game, level, output=False)
vrle.immovables = immovables




