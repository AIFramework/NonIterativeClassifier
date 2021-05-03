using AI;
using AI.DataStructs;
using AI.ML.AlgorithmAnalysis;
using AI.ML.Classifiers;
using AI.ML.Datasets;
using AI.ML.NeuralNetwork.CoreNNW.Layers;
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
            classifier = new SimpleClassifier(3, 4); // Создание классификатора, 3 класса, 4 признака

            var dataset = GetIrisDataset(); // Получение датасета
            Shuffling<VectorClass>.Shuffle(dataset); // Равномерное перемешивание
            int len = dataset.Length, len25 = len / 4, len75 = len - len25;
            // Обучающая / тестовая выборка
            var test = new VectorClass[len75]; 
            var train = new VectorClass[len25];
            for (int i = 0; i < len75; i++) test[i] = dataset[i];
            for (int i = 0; i < len25; i++) train[i] = dataset[i+len75];


            VectorIntDataset vectorClasses = new VectorIntDataset();
            vectorClasses.AddRange(train); // Добавление обучающей выборки
            Stopwatch stopwatch = new Stopwatch();


            stopwatch.Start(); // Цикл на 100 итераций нужен только для точности замера времени, в реальной работе следует его убрать
            for (int i = 0; i < 100; i++) classifier.Train(vectorClasses); // Обучение
            stopwatch.Stop();
            Console.WriteLine($"Время обучения: {stopwatch.ElapsedMilliseconds / (1000.0 * 100.0)} сек");

            //classifier.Train(vectorClasses); // Обучение

            Console.WriteLine("=====================================================".ToUpper());
            Console.WriteLine("\n\n===== Безытеративный классификатор ===== \n\n\n".ToUpper());
            Console.WriteLine("=====================================================".ToUpper());
            
            Test(test); // Тестирование



            Console.WriteLine("\n\n\nКонвертирование в нейросеть".ToUpper());
            Console.WriteLine("=====================================================".ToUpper());
            classifier = new NeuralClassifier((classifier as SimpleClassifier).GetNetwork() as NNW) { EpochNum = 3};
            Test(test); // Тестирование


            Console.WriteLine("\n\n\n Тюнинг методом Adam".ToUpper());
            Console.WriteLine("=====================================================".ToUpper());
            Console.WriteLine("\n\n");
            classifier.Train(vectorClasses);
            Console.WriteLine("\n\n");
            Test(test); // Тестирование
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
