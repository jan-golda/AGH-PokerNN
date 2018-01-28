module NeuralNetwork.Training (TrainingCase(TrainingCase, input, expected), TrainingSet, trainOnCase, trainOnSet, trainingSetShuffle, epochTraining) where

  import NeuralNetwork

  import System.Random.Shuffle as Shuffle
  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------

  data TrainingCase = TrainingCase {input :: Matrix Double, expected :: Matrix Double} deriving (Show)

  type TrainingSet = [TrainingCase]

  ---------------------------------------------------------------------------------
  -- FUNCTIONS
  ---------------------------------------------------------------------------------

  -- | Trains network on a single training case
  trainOnCase :: NeuralNetwork -> TrainingCase -> Double -> NeuralNetwork
  trainOnCase network trainingCase learningRate = NeuralNetwork.learn (input trainingCase) (expected trainingCase) learningRate network

  -- | Trains network on given dataset
  trainOnSet :: NeuralNetwork -> TrainingSet -> Double -> NeuralNetwork
  trainOnSet network [] _ = network
  trainOnSet network (x:xs) learningRate = trainOnSet (trainOnCase network x learningRate) xs learningRate

  -- | Shuffles training set
  trainingSetShuffle :: TrainingSet -> IO TrainingSet
  trainingSetShuffle = Shuffle.shuffleM

  -- | ???
  epochTraining :: NeuralNetwork -> TrainingSet -> Double -> Int -> NeuralNetwork
  epochTraining network _ _ 0 = network
  epochTraining network trainingSet learningRate epochsNumber = epochTraining newNetwork trainingSet learningRate (epochsNumber - 1)
    where newNetwork = trainOnSet network trainingSet learningRate


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
