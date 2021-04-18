﻿using AI;
using AI.DataStructs;
using AI.ML.AlgorithmAnalysis;
using AI.ML.Classifiers;
using AI.ML.Datasets;
using NonIterative;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace NonIterativeTest
{
    class Program
    {
        static IClassifier classifier;

        static void Main(string[] args)
        {
            classifier = new SimpleClassifier(3, 4);

            var dataset = GetIrisDataset();
            Shuffling<VectorClass>.Shuffle(dataset);
            int len = dataset.Length, len25 = len / 4, len75 = len - len25;
            // Обучающая / тестовая выборка
            var test = new VectorClass[len75];
            var train = new VectorClass[len25];
            for (int i = 0; i < len75; i++) test[i] = dataset[i];
            for (int i = 0; i < len25; i++) train[i] = dataset[i+len75];


            VectorIntDataset vectorClasses = new VectorIntDataset();
            vectorClasses.AddRange(train);
            Stopwatch stopwatch = new Stopwatch();


            stopwatch.Start(); // Цикл на 1000 итераций нужен только для точности замера времени, в реальной работе следует его убрать
            for(int i = 0; i<1000; i++) classifier.Train(vectorClasses);
            stopwatch.Stop();
            Console.WriteLine($"Время обучения: {stopwatch.ElapsedMilliseconds /(1000.0* 1000.0)} сек");

            Console.WriteLine("=====================================================".ToUpper());
            Console.WriteLine("\n\n===== Безытеративный классификатор ===== \n\n\n".ToUpper());
            Console.WriteLine("=====================================================".ToUpper());
            
            Test(test);
        }

        static VectorClass[] GetIrisDataset(string path = "iris.txt") 
        {
            string[] objects = File.ReadAllLines(path);
            var dataset = new List<VectorClass>(objects.Length);
            var provider = AISetting.GetProvider();

            foreach (var item in objects)
            {
                var strs = item.Split(',');
                double[] features = new double[4];
                

                for (int i = 0; i < 4; i++) features[i] = Convert.ToDouble(strs[i], provider);

                int mark = -1;

                switch (strs[4]) 
                {
                    case "\"Setosa\"":
                        mark = 0;
                        break;

                    case "\"Versicolor\"":
                        mark = 1;
                        break;

                    case "\"Virginica\"":
                        mark = 2;
                        break;
                }


                dataset.Add(new VectorClass(features, mark));
            }

            return dataset.ToArray();
        }

        static void Test(VectorClass[] test)
        {
            int[] ideal = new int[test.Length], outp = new int[test.Length];

            for (int i = 0; i < test.Length; i++)
            {
                ideal[i] = test[i].ClassMark;
                outp[i] = classifier.Classify(test[i].Features);
            }

           Console.WriteLine(MetricsForClassification.FullReport(ideal, outp, isForEachClass:true));
        }
    }
}