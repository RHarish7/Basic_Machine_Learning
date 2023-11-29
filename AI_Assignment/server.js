const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
var cors = require("cors");

const app = express();
const port = 5000;

app.use(cors());

app.use(bodyParser.json());

app.post(`/api/chat`, async (req, res) => {
  const { message } = req.body;

  try {
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-3.5-turbo",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: message },
        ],
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization:
            "Bearer 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" /*  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 */,
        },
      }
    );

    const reply = response.data.choices[0].message.content;
    res.json({ reply });
  } catch (error) {
    console.error(
      "Error in API request to OpenAI:",
      error.response?.data || error.message
    );
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
