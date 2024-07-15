import { config } from "dotenv";
import { HfInference } from "@huggingface/inference";
import { promises as fs } from 'fs';

// Load environment variables from .env file
config();

// Get the API key from the environment variable
const HF_TOKEN = process.env.HF_TOKEN;

if (!HF_TOKEN) {
    throw new Error("HF_TOKEN is not defined in the environment variables");
}

const inference = new HfInference(HF_TOKEN);

async function main() {
    // Chat completion API
const out = await inference.chatCompletion({
    model: "mistralai/Mistral-7B-Instruct-v0.2",
    messages: [{ role: "user", content: "Complete the this sentence with words one plus one is equal " }],
    max_tokens: 100
  });
  console.log(out.choices[0].message);

  const data = await inference.textToImage({
    model: 'stabilityai/stable-diffusion-2',
    inputs: 'award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]',
    parameters: {
      negative_prompt: 'blurry',
    }
  })

    // Convert Blob to ArrayBuffer
    const arrayBuffer = await data.arrayBuffer();

    // Convert ArrayBuffer to Buffer
    const buffer = Buffer.from(arrayBuffer);

    // Determine file extension based on content type (e.g., 'image/jpeg')


  await fs.writeFile("data.jpg", buffer);
  console.log(`Image saved to ${"data.jpg"}`);
  
  console.log(data)

}

main().catch(console.error);