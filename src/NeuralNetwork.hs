module NeuralNetwork (NeuralNetwork, Layer, networkOutput) where

  import Data.Matrix as Matrix

  ---------------------------------------------------------------------------------
  -- TYPE DEFINITIONS
  ---------------------------------------------------------------------------------
  data Layer = Layer {weights :: Matrix Double, biases :: Matrix Double} deriving (Show)

  type NeuralNetwork = [Layer]

  ---------------------------------------------------------------------------------
  -- DEFINITIONS
  ---------------------------------------------------------------------------------
  layerOutput :: Layer -> Matrix Double -> Matrix Double
  networkOutput :: NeuralNetwork -> Matrix Double -> Matrix Double

  sigmoid :: Double -> Double

  weightedInput :: Layer -> Matrix Double -> Matrix Double

  ---------------------------------------------------------------------------------
  -- IMPLEMENTATION
  ---------------------------------------------------------------------------------
  layerOutput layer input = Matrix.mapCol (\_ x -> sigmoid x) 1 (weightedInput layer input)

  networkOutput []     input = input
  networkOutput (l:ls) input = networkOutput ls (layerOutput l input)
  
  sigmoid x = 1/(1 + exp (-x))

  weightedInput layer input = Matrix.elementwise (+) (Matrix.multStd (weights layer) input) (biases layer)
