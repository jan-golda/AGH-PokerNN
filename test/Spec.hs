
import Test.HUnit

import Data.Matrix as Matrix

import NeuralNetwork
import NeuralNetwork.IO

-- run tests
main :: IO Counts
main = runTestTT (TestList [nnTests, ioTests])

-- NeuralNetworks used during tests
nn1 = [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11])]
nn2 = [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77])]
nn3 = [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77]), Layer (Matrix.fromList 1 3 [1.1, 2.2, 3.3]) (Matrix.fromList 1 1 [322.0])]

---------------------------------------------------------------------------------
-- HUnit: NeuralNetwor
---------------------------------------------------------------------------------
nnTests = TestList $ [
          TestLabel "Feed 1L NN #1" testFeed1,
          TestLabel "Feed 1L NN #2" testFeed2,
          TestLabel "Feed 1L NN #3" testFeed3,
          TestLabel "Feed 2L NN #1" testFeed4,
          TestLabel "Feed 2L NN #2" testFeed5,
          TestLabel "Feed 2L NN #3" testFeed6,
          TestLabel "Feed 3L NN #1" testFeed7,
          TestLabel "Feed 3L NN #2" testFeed8,
          TestLabel "Feed 3L NN #3" testFeed9]

testFeed1 :: Test
testFeed1 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 2 1 [0.9999999866399913,0.9121360851706988])
      (feed nn1 (Matrix.fromList 3 1 [0.4, 0.9, 0.0]))

testFeed2 :: Test
testFeed2 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 2 1 [0.9999999999994043,0.8205384805926733])
      (feed nn1 (Matrix.fromList 3 1 [2.4, -0.9, 0.0]))

testFeed3 :: Test
testFeed3 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 2 1 [0.9809228026847419,0.9999451031670289])
      (feed nn1 (Matrix.fromList 3 1 [0.0, 0.0, 1.0]))

testFeed4 :: Test
testFeed4 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 3 1 [0.9950331982178893,0.999999997254764,0.9999999997616016])
      (feed nn2 (Matrix.fromList 3 1 [0.4, 0.9, 0.0]))

testFeed5 :: Test
testFeed5 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 3 1 [0.9950331983499373,0.9999999958921633,0.9999999996798188])
      (feed nn2 (Matrix.fromList 3 1 [2.4, -0.9, 0.0]))

testFeed6 :: Test
testFeed6 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 3 1 [0.994841027931582,0.999999998109462,0.9999999997764508])
      (feed nn2 (Matrix.fromList 3 1 [0.0, 0.0, 1.0]))

testFeed7 :: Test
testFeed7 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 1 1 [1.0])
      (feed nn3 (Matrix.fromList 3 1 [0.4, 0.9, 0.0]))

testFeed8 :: Test
testFeed8 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 1 1 [1.0])
      (feed nn3 (Matrix.fromList 3 1 [2.4, -0.9, 0.0]))

testFeed9 :: Test
testFeed9 = TestCase $ assertEqual "NN wrongly calculated output"
      (Matrix.fromList 1 1 [1.0])
      (feed nn3 (Matrix.fromList 3 1 [0.0, 0.0, 1.0]))

---------------------------------------------------------------------------------
-- HUnit: NeuralNetwork.IO
---------------------------------------------------------------------------------
ioTests = TestList $ [
          TestLabel "Read 1L NN" testFromString1,
          TestLabel "Read 2L NN" testFromString2,
          TestLabel "Read 3L NN" testFromString3,
          TestLabel "Write 1L NN" testToString1,
          TestLabel "Write 2L NN" testToString2,
          TestLabel "Write 3L NN" testToString3]

testFromString1 :: Test
testFromString1 = TestCase $ assertEqual "NN wrongly readed from string"
      nn1
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n")

testFromString2 :: Test
testFromString2 = TestCase $ assertEqual "NN wrongly readed from string"
      nn2
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n")

testFromString3 :: Test
testFromString3 = TestCase $ assertEqual "NN wrongly readed from string"
      nn3
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n1.1 2.2 3.3\n322\n")

testToString1 :: Test
testToString1 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n"
      (toString nn1)

testToString2 :: Test
testToString2 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n"
      (toString nn2)

testToString3 :: Test
testToString3 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n1.1 2.2 3.3\n322.0\n"
      (toString nn3)
