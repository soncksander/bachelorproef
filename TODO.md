# TODO

## Database

- een init.sql file maken met de database table creation scripts en het path hiernaartoe aanpassen in de !(docker compose file)[./compose.yml]
- (Gebruik type VECTOR(dimensie) voor de column die de embeddings opslaat)
- maak een .env file aan met de volgende velden:

```
POSTGRES_USER=sander
POSTGRES_PASSWORD=
POSTGRES_DB=<naam_van_database>
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Daarna kan je docker compose up -d --build doen en de database zal automatisch geinitialiseerd worden

## Recepten parsing

Flow maken waarin je een LLM de recepten geeft, de AI doet een toolcall / structured output en geeft de geparste recepten terug

- Opzoeken hoe toolcalls / structured output worden gedaan, zie daarvoor !(OpenRouter docs)[[openrouter.ai/docs](https://openrouter.ai/docs/quickstart)]
(structured output lijkt me iets beter in jouw geval)
- Flow opzetten: inlezen data -> LLM -> uitschrijven van nieuwe data
- Qua cost/performance raad ik Gemini 2.5-flash-lite, GPT-5-nano of Grok-4-Fast aan (moet je maar eens testen welke het het best doet of nog andere models kiezen en testen)
