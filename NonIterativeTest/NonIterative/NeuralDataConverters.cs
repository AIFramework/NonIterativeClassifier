using AI.DataStructs.Algebraic;
using AI.ML.NeuralNetwork.CoreNNW;
using System;

namespace NonIterative
{
    public class NeuralDataConverters
    {
        public static NNValue WVectorsToNNValue(Vector[] matrW)
        {
            if (matrW.Length == 0)
            {
                throw new Exception("Matrix w is empty");
            }

            int h = matrW.Length, w = matrW[0].Count;

            NNValue matrixW = new NNValue(h, w);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    matrixW[i, j] = (float)matrW[i][j];
                }
            }

            return matrixW;
        }
    }
}
