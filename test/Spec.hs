
import Test.HUnit

import Data.Matrix as Matrix

import NeuralNetwork
import NeuralNetwork.IO

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
      [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11])]
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n")

testFromString2 :: Test
testFromString2 = TestCase $ assertEqual "NN wrongly readed from string"
      [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77])]
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n")

testFromString3 :: Test
testFromString3 = TestCase $ assertEqual "NN wrongly readed from string"
      [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77]), Layer (Matrix.fromList 1 3 [1.1, 2.2, 3.3]) (Matrix.fromList 1 1 [322.0])]
      (fromString "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n1.1 2.2 3.3\n322\n")

testToString1 :: Test
testToString1 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n"
      (toString [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11])])

testToString2 :: Test
testToString2 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n"
      (toString [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77])])

testToString3 :: Test
testToString3 = TestCase $ assertEqual "NN wrongly writed to string"
      "14.0 9.99 0.4 1.3 1.9 9.7\n3.54 0.11\n2.0 0.0 0.7 4.4 11.45 3.22\n3.3 15.0 7.77\n1.1 2.2 3.3\n322.0\n"
      (toString [Layer (Matrix.fromList 2 3 [14.0, 9.99, 0.4, 1.3, 1.9, 9.7]) (Matrix.fromList 2 1 [3.54, 0.11]), Layer (Matrix.fromList 3 2 [2.0, 0.0, 0.7, 4.4, 11.45, 3.22]) (Matrix.fromList 3 1 [3.3, 15.0, 7.77]), Layer (Matrix.fromList 1 3 [1.1, 2.2, 3.3]) (Matrix.fromList 1 1 [322.0])])

main :: IO Counts
main = runTestTT ioTests
