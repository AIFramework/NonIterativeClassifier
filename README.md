# NonIterativeClassifier
Классификатор с безытеративным обучением

* Безытеративное обучение, это обучение классификатора за одну итерацию, прямой расчет весовых коэффициентов

Этой темой я занимаюсь довольно давно, написал 3 статьи по ней, 2 на хабр, ссылки ниже, одну в рамках студ. конференции, занял с этим докладом первое место. Также эта стратегия обучения была основой моей бакалаврской работы, в которой я классифицировал сигналы. Решил, что необходимо опубликовать сжатую теорию по этой теме и код. 

Мотивация заключается в следующем, если мы сможем обучать нейронные сети безытеративно, то сможем это делать в разы быстрей.

Статьи на Хабр: 
 * [Метод безытеративного обучения однослойной сети прямого распространения с линейной активационной функцией](https://habr.com/ru/post/332936)
 * [Безытеративное обучение однослойного персептрона. Задача классификации](https://habr.com/ru/post/333382)


## Тест

* Датасет: "Ирисы Фишера"
* Тренировочная выборка: 25%
* Тестовая выборка: 75%
* Время обучения: 0,000272
* F1 мера: 82,67%

## Код теста

```c#

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


            stopwatch.Start();
            for(int i = 0; i<1000; i++) classifier.Train(vectorClasses);
            stopwatch.Stop();
            Console.WriteLine($"Время обучения: {stopwatch.ElapsedMilliseconds /(1000.0* 1000.0)} сек");

            Console.WriteLine("=====================================================".ToUpper());
            Console.WriteLine("\n\n===== Безытеративный классификатор ===== \n\n\n".ToUpper());
            Console.WriteLine("=====================================================".ToUpper());
            
            Test(test);
```

## Полная информация о тесте

``` 
Время обучения: 0,000272 сек
=====================================================


===== БЕЗЫТЕРАТИВНЫЙ КЛАССИФИКАТОР =====



=====================================================
Precision:             0,8068   80,68%
Average Recall:      0,8476     84,76%
FMeasure:            0,8267     82,67%
Accuracy:            0,8053     80,53%


--Precision value for each class--

Class #1:  0,9737       97,37%
Class #2:  0,4737       47,37%
Class #3:  0,9730       97,30%

```
