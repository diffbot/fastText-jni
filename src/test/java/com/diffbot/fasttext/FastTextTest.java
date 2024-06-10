package com.diffbot.fasttext;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

public class FastTextTest {

    @Test
    public void modelTest() throws IOException {
        FastTextModel model;
        /*
         * language identification
         * https://fasttext.cc/docs/en/language-identification.html
         */
        try(InputStream inputStream = this.getClass().getResourceAsStream("lid.176.ftz")) {
            model = new FastTextModel(inputStream);
        }

        {
            Prediction prediction = model.predictProba("Web Data for your AI Imagine if your app could access the web like a structured database .");
            System.out.println(prediction.label + " : " + prediction.probability);
            Assertions.assertEquals("__label__en", prediction.label);
        }

        {
            Prediction prediction = model.predictProba("AI のための Web データ アプリが構造化データベースのように Web にアクセスできるかどうかを想像してください。");
            System.out.println(prediction.label + " : " + prediction.probability);
            Assertions.assertEquals("__label__ja", prediction.label);
        }

        {
            Prediction[] predictions = model.predictProbaTopK(" ", -1);
            Assertions.assertEquals(176, predictions.length);
        }

        {
            float threshold = 0.01f;
            Prediction[] predictions = model.predictProbaWithThreshold(" ", threshold);
            Assertions.assertEquals(24, predictions.length);
            double prev = Float.MAX_VALUE;
            for (Prediction prediction : predictions) {
                Assertions.assertTrue(prediction.probability >= threshold);
                Assertions.assertTrue(prediction.probability <= prev);
                prev = prediction.probability;
            }
        }

        model.close();

        /*
         * https://fasttext.cc/docs/en/supervised-models.html
         * World (1), Sports (2), Business (3), Sci/Tech (4)
         */
        try(InputStream inputStream = this.getClass().getResourceAsStream("ag_news.ftz")) {
            model = new FastTextModel(inputStream);
        }

        Prediction prediction = model.predictProba("web data for your ai imagine if your app could access the web like a structured database .");
        System.out.println(prediction.label + " : " + prediction.probability);
        Assertions.assertEquals("__label__4", prediction.label);


        Prediction[] array = model.predictProbaTopK("web data for your ai imagine if your app could access the web like a structured database .", 5);

        for (int i = 0; i < array.length; i++) {
            Prediction p = array[i];
            System.out.println(p.label + " : " + p.probability);
        }

        Assertions.assertEquals(4, array.length);

        model.close();
    }

}
