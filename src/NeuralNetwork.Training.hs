module NeuralNetwork.Training (TrainingCase, TrainingSet, intListToTrainingCase, train) where

  import Data.Matrix as Matrix
  import NeuralNetwork 

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------
  data TrainingCase = TrainingCase {input :: Matrix Double, expected :: Matrix Double} deriving (Show)

  type TrainingSet = [TrainingCase]


  ---------------------------------------------------------------------------------
  -- FUNCTIONS
  ---------------------------------------------------------------------------------
  
  -- Trains network on a single training case
                                              
  singleCaseTrain :: NeuralNetwork -> TrainingCase -> Double -> NeuralNetwork
  singleCaseTrain network trainingCase learningRate = NeuralNetwork.learn (input trainingCase) (expected trainingCase) learningRate network
  
  -- Trains network on given dataset
  
  train :: NeuralNetwork -> TrainingSet -> Double -> NeuralNetwork
  train network [] _ = network
  train network (x:xs) learningRate = train (singleCaseTrain network x learningRate) xs learningRate
  
  
  
  {-
  -- Int list representation of test case input:
  -- "1s,1r,2s,2r,3s,3r,4s,4r,5s,5r"
  -- where:
  --    ks - suit of k-th card (1 - Hearts, 2 - Spades, 3 - Diamonds, 4 - Clubs)
  --    kr - rank of k-th card (1 - Ace, 2 - 2, 3 - 3, ..., 12 - Queen, 13 - King)
  --
  -- Int representation of test case output is a single digit representing hands as follows:
  -- 0: Nothing in hand (high card); not a recognized poker hand 
  -- 1: One pair; one pair of equal ranks within five cards 
  -- 2: Two pairs; two pairs of equal ranks within five cards 
  -- 3: Three of a kind; three equal ranks within five cards 
  -- 4: Straight; five cards, sequentially ranked with no gaps 
  -- 5: Flush; five cards with the same suit 
  -- 6: Full house; pair + different rank three of a kind 
  -- 7: Four of a kind; four equal ranks within five cards 
  -- 8: Straight flush; straight + flush 
  -- 9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush 
  --
  -- Matrix representation of input is a single column matrix with 52 rows, each row representing presence (1) or absence (0) of a card in our 5-card hand
  -- First 13 rows represent Hearts, next 13 - Spades, next - Diamonds and finally Clubs
  --
  -- Matrix representation of output is a singla column matrix with 10 rows, all zeros except for row corresponding with proper hand strength
  -}
  
  
  -- Converts Int list TestCase input representation to proper Matrix compatible with NeuralNetwork input
  
  intListToMatrixInput :: [Int] -> Matrix Double
  intListToMatrixInput intListInput = 
      let
          suits = map fst ( filter (\x -> even (snd x)) (zip intListInput [0..9]) )
          ranks = map fst ( filter (\x -> odd (snd x)) (zip intListInput [0..9]) )
          cards = [(suit - 1) * 13 + (rank - 1) | (suit, rank) <- zip suits ranks]
          presenceList = map (\x -> if x `elem` cards then 1.0 else 0.0) (take 52 [0..])
      in
          Matrix.fromList 52 1 presenceList
              
        
  
  -- Converts Int TestCase output respresentation to Matrix representing network's expected output
  
  intToMatrixOutput :: Int -> Matrix Double
  intToMatrixOutput intOutput =
      let 
          handsList = map (\x -> if x == intOutput then 1.0 else 0.0) (take 10 [0..])
      in
          Matrix.fromList 10 1 handsList
          
  
  -- Converts list of Ints (first 11 elements) to single TrainingCase
  
  intListToTrainingCase :: [Int] -> TrainingCase
  intListToTrainingCase intList = TrainingCase (intListToMatrixInput intListInput) (intToMatrixOutput intOutput)
      where
          intListInput = take 10 intList
          intOutput = head.reverse.(take 11) $ intList