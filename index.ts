import { LLMChain, VectorDBQAChain } from "langchain/chains";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAI } from "langchain/llms/openai";
import { PromptTemplate } from "langchain/prompts";
import readline from "readline-sync";

import { PineconeClient } from "@pinecone-database/pinecone";
import * as dotenv from "dotenv";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { CreateRequest } from "@pinecone-database/pinecone";

dotenv.config();

const model = new OpenAI({ openAIApiKey: process.env.OPEN_AI_KEY, temperature: 0 })

async function callingOpenAIWithoutPrompt() {
    try {
        do {
            const response = await model.call("What is prompt engineering?")
            console.log(response)
        } while (['y', 'yes'].includes(readline.question('Do you want to try again? ').toLowerCase()))
    } catch (error) {
        console.error(error)
    }
}

async function explainInLaymenTerms() {
    try {
        do {
            const subject = readline.question("Hi!, I can help you understand any topic in simple words. Tell me what would you like me to help you understand better? ")
            const template = `Explain me in layman's terms the following topic which is in double quotes "{subject}"`;
            const prompt = new PromptTemplate({
                template,
                inputVariables: ['subject']
            })
            const chain = new LLMChain({ llm: model, prompt })
            const response = await chain.call({ subject })
            console.log(response.text)
        } while (['y', 'yes'].includes(readline.question('Anything else? ').toLowerCase()))
    } catch (error) {
        console.error(error)
    }
}

async function pdfLoader(path: string) {
    try {
        const loader = new PDFLoader(path, {
            splitPages: false,
        });
        const docs = await loader.load();
        return docs[0].pageContent
    } catch (error) {
        console.error(error)
    }
}

async function createPinconeIndexAndUploadData() {
    try {
        const indexName = 'uniblox-app-status'
        const client = new PineconeClient();
        await client.init({
            apiKey: process.env.PINECONE_API_KEY as string,
            environment: process.env.PINECONE_ENVIRONMENT as string,
        });
        console.log('Getting all indexes')
        const allIndexes = await client.listIndexes()
        console.log('Indexes', allIndexes)
        if (!allIndexes.includes(indexName)) {
            const createRequest: CreateRequest = {
                name: indexName,
                dimension: 1536
            }
            const createIndex = await client.createIndex({ createRequest })
        }
        const pineconeIndex = client.Index(indexName);

        const data = await pdfLoader("uniblox-app-status.pdf")
        const docs = [
            new Document({
                metadata: { title: 'uniblox-application-status' },
                pageContent: data as string,
            })
        ];

        await PineconeStore.fromDocuments(docs, new OpenAIEmbeddings({openAIApiKey: process.env.OPEN_AI_KEY}), {
            pineconeIndex,
        });

    } catch (error) {
        console.error(error)
    }
}

async function getEmbeddingsFromOpenAI (text: string) {
    try {
        const embeddingModel = new OpenAIEmbeddings({
            openAIApiKey: process.env.OPEN_AI_KEY
        })
        const embedding = await embeddingModel.embedQuery(text)
        console.log(embedding)
        return embedding
    } catch (error) {
        console.error(error)
    }
}

async function queryWithCustomKnowledge (query: string) {
    try {
        const indexName = 'uniblox-app-status'
        const client = new PineconeClient();
        await client.init({
          apiKey: process.env.PINECONE_API_KEY as string,
          environment: process.env.PINECONE_ENVIRONMENT as string,
        });
        const pineconeIndex = client.Index(indexName);
        const vectorStore = await PineconeStore.fromExistingIndex(
            new OpenAIEmbeddings({openAIApiKey: process.env.OPEN_AI_KEY}),
            { pineconeIndex }
          );
          const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
            k: 1,
            returnSourceDocuments: true,
          });
          const response = await chain.call({ query });
          return response.text.trim()
    } catch (error) {
        console.error(error)
    }
}


async function main() {
    try {
        // console.log('Welcome, I am here to help you with your uniblox application related queries')
        // const exitInput = ['N', 'n']
        // let userInput;
        // do {
        //     userInput = readline.question('Ask me a question (Press "N" to exit else type question)? ')
        //     if (!exitInput.includes(userInput.trim())) {
        //         const response = await queryWithCustomKnowledge(userInput)
        //         console.log(response)
        //         console.log('\n')
        //     }
        // } while (!exitInput.includes(userInput.trim()))

        console.log(await pdfLoader('uniblox-app-status.pdf'))
    } catch (error: Error | any) {
        console.error(error)
    }
}

main()