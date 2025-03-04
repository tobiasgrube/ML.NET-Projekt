using System;
using Microsoft.ML;
using Microsoft.ML.Data;

// 1️ Datenklasse für Kundenbewertungen
public class ReviewData
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1)]
    public bool Sentiment { get; set; } // true = positiv, false = negativ
}

// 2️ Ausgabeformat des ML-Modells
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; } 
}

class Program
{
    static void Main()
    {
        // 3️ ML-Kontext erstellen
        var mlContext = new MLContext();

        // 4️ Trainingsdaten vorbereiten
        var data = new[]
        {
            new ReviewData { Text = "Das Produkt ist fantastisch!", Sentiment = true },
            new ReviewData { Text = "Ich liebe diesen Artikel.", Sentiment = true },
            new ReviewData { Text = "Ich bin enttäuscht von der Qualität.", Sentiment = false },
            new ReviewData { Text = "Nie wieder! Schrecklicher Kauf.", Sentiment = false }
        };

        var trainData = mlContext.Data.LoadFromEnumerable(data);

        // 5️ Machine-Learning-Pipeline aufbauen
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text")
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Sentiment", featureColumnName: "Features"));

        // 6️ Modell trainieren
        var model = pipeline.Fit(trainData);

        // 7️ Modell testen
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ReviewData, SentimentPrediction>(model);

        // 8️ Beispiel-Vorhersage
        Console.Write("Gib eine Bewertung ein: ");
        string inputText = Console.ReadLine();
        Console.WriteLine("KI analysiert die Bewertung...");
        var inputData = new ReviewData { Text = inputText };
        var prediction = predictionEngine.Predict(inputData);

        // 9️ Ergebnis anzeigen
        Console.WriteLine($"Das System glaubt, die Bewertung ist: {(prediction.Prediction ? "Positiv 😊" : "Negativ 😡")}");
    }
}
