using App.Terminal.Entities;
using Microsoft.ML;
using System;

namespace App.Terminal
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            /* PASO 1) Importar la informacion */

            HouseData[] houseData = {
               new HouseData() { Size = 1.1F, Price = 1.2F },
               new HouseData() { Size = 1.9F, Price = 2.3F },
               new HouseData() { Size = 2.8F, Price = 3.0F },
               new HouseData() { Size = 3.4F, Price = 3.7F } };

            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            /* PASO 2) Especificar la informacion preparada*/

            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" }).Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            /* PASO 3) Entrenar el Modelo*/

            var model = pipeline.Fit(trainingData);

            /* PASO 4) Realizar la prediccion */

            var size = new HouseData() { Size = 2.5F };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size*1000} sq ft= {price.Price*100:C}k");
            Console.ReadLine();

        }
    }
}
