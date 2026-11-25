# Terms of use

These terms cover your use of this server and of the AI models it runs. The operator of this
server, the person or organisation that runs it, sets them and is responsible for them.

> **For the operator.** This page is a template that comes with Brilliance. It is not legal
> advice, and the Brilliance and Radiance projects are not a party to it. Before you let other
> people use your server, adopt it as your own: copy it, name yourself as the operator, remove the
> models you do not serve, add your own terms, and set `terms-file:` under `server:` to your copy.
> Then remove this note. Some model licences require whoever serves the model to other people to
> pass the licence's use restrictions on to them as enforceable terms. The sections below carry
> those restrictions for the models Brilliance 0.3.1 lists. Other licences set conditions on you,
> the operator, before you serve the model commercially: Qwen3.8-Flash-Next's licence is one
> (its section below). Read those conditions before you serve the model to anyone.

## Using this service

- By using this service you agree to these terms, including the section below for each model you
  use.
- You are responsible for what you send to the service, for what it makes for you, and for how you
  use what it makes.
- Follow the laws that apply to you and to your use of the service.
- The operator may change these terms. The version on this page is the one that applies.

## The models and their licences

Each model keeps the licence its makers gave it. Where a licence restricts how the model and what
it makes may be used, those restrictions are part of these terms whenever you use that model here.
The full text of each licence is at `/terms/licenses` on this server.

| Model | Licence | Use restrictions |
|---|---|---|
| LTX-2.5 distilled (video) | LTX-2.x Community License Agreement, Lightricks Ltd. | Yes, below |
| SDXL base 1.0 (pictures) | CreativeML Open RAIL++-M | Yes, below |
| EmbeddingGemma 2 (embeddings) | Apache 2.0 | The Gemma Prohibited Use Policy, below |
| Qwen3.8-27B | Apache 2.0 | None |
| Qwen3.8-Flash-Next | Qwen Community License 1.0 | Yes, below: lawful use that infringes no one's intellectual property; commercial-use and attribution conditions on the operator |
| MiniCPM5-2B | Apache 2.0 | None |
| Inkling 2 Mini | Apache 2.0 | None |
| Qwen3-ASR 1.7B (transcription) | Apache 2.0 | None |
| Nemotron 3.5 Rerank 8B (rerank) | OpenMDW 1.1 | None |
| Gemma 4 31B and 12B | Apache 2.0 | None |
| Qwen3.6-35B-A3B | Apache 2.0 | None |
| Qwen-Image 2.1 (pictures) | Qwen RESEARCH LICENSE AGREEMENT | Yes, below: non-commercial use only |
| DeepSeek-V4.1-Flash-1017 | MIT | None |
| Hy4-flash | Tencent Hy License 1.0 | Yes, below: a separate licence for large Model-as-a-Service businesses; the licence and notices kept with copies |

## LTX-2.5 (video)

LTX-2.5 is licensed under the LTX-2.x Community License Agreement of Lightricks Ltd. (licence date
August 11, 2026). LTX-2.5 and the videos it makes are subject to Section 4 and Attachment A of that
licence in their entirety, including all use restrictions and acceptable use policies. By using
LTX-2.5 here you agree to comply with them, and they are an enforceable part of these terms. They
are quoted below as the licence states them. Words the licence defines, such as LTX-2.x, Outputs,
Derivatives and Harm, have the meanings its section 1 gives them. The whole licence is at
`/terms/licenses/ltx-2.x-community-license` on this server, and in the model's repository,
https://huggingface.co/StillDeadcode/radiance-ltx-2.5-distilled-iq8r.

Section 4 of the licence:

```text
   4. Use-based restrictions. The restrictions set forth in Attachment A
      are considered Use-based restrictions. Therefore, you cannot use
      LTX-2.x and the Derivatives of LTX-2.x in violation of the
      specified restricted uses. You may use LTX-2.x subject to this
      Agreement, only for lawful purposes and in accordance with the
      Agreement. "Use" may include creating any content with,
      fine-tuning, updating, running, training, evaluating and/or
      re-parametrizing LTX-2.x. You shall require all of your users who
      use LTX-2.x or a Derivative of LTX-2.x to comply with the terms of
      this Section 4.
```

Attachment A of the licence, which includes Lightricks' Acceptable Use Policy by its address:

```text
                              Attachment A
                            Use Restrictions

   When using the Outputs, LTX-2.x and any Derivatives thereof, you
   agree to comply with the Acceptable Use Policy
   (https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf)
   which is hereby incorporated into and made part of this Agreement by
   reference. Licensor may update it from time to time, and the version
   in effect at the time of your use governs; continued use after an
   update constitutes acceptance. Licensor shall post each version of
   the Acceptable Use Policy with its effective date, and no update
   shall apply retroactively to use occurring before that effective
   date. In addition, you agree not to use the Outputs, LTX-2.x or its
   Derivatives in any of the following ways:

      1) In any way that violates any applicable national, federal,
         state, local or international law or regulation;

      2) For the purpose of exploiting, Harming or attempting to exploit
         or Harm minors in any way;

      3) Knowingly generate or disseminate verifiably false information
         and/or content with the intent to deceive, defraud, or
         otherwise unlawfully Harm others;

      4) To generate or disseminate personal identifiable information
         that can be used to Harm an individual;

      5) To generate or disseminate information and/or content (e.g.
         images, code, posts, articles), and place the information
         and/or content in any context (e.g. bot generating tweets)
         without expressly and intelligibly disclaiming that the
         information and/or content is machine generated;

      6) To defame others, or to engage in the unlawful harassment of
         others;

      7) To impersonate or attempt to impersonate (e.g. deepfakes)
         others without their consent;

      8) For fully automated decision making that adversely impacts an
         individual's legal rights or otherwise creates or modifies a
         binding, enforceable obligation;

      9) For any use intended to or which has the effect of
         discriminating against or Harming individuals or groups based
         on online or offline social behavior or known or predicted
         personal or personality characteristics;

      10) To exploit any of the vulnerabilities of a specific group of
          persons based on their age, social, physical or mental
          characteristics, in order to materially distort the behavior
          of a person pertaining to that group in a manner that causes
          or is likely to cause that person or another person physical
          or psychological Harm;

      11) For any use intended to or which has the effect of
          discriminating against individuals or groups based on legally
          protected characteristics or categories;

      12) To provide medical advice and medical results interpretation;

      13) To generate or disseminate information for the purpose to be
          used for administration of justice, law enforcement,
          immigration or asylum processes, such as predicting an
          individual will commit fraud/crime commitment (e.g. by text
          profiling, drawing causal relationships between assertions
          made in documents, indiscriminate and arbitrarily-targeted
          use);

      14) To generate and/or disseminate malware (including - but not
          limited to - ransomware) or any other content to be used for
          the purpose of harming electronic systems;

      15) To engage in, promote, incite, or facilitate discrimination or
          other unlawful or harmful conduct in the provision of
          employment, employment benefits, credit, housing, or other
          essential goods and services;

      16) To engage in, promote, incite, or facilitate the harassment,
          abuse, threatening, or bullying of individuals or groups of
          individuals;

      17) For military, warfare, nuclear industries or applications,
          weapons development, or any use in connection with activities
          that may cause death, personal injury, or severe physical or
          environmental damage;

      18) For commercial use only: To train, improve, or fine-tune any
          other machine learning model, artificial intelligence system,
          or competing model, except for Derivatives of LTX-2.x as
          expressly permitted under this Agreement;

      19) To circumvent, disable, or interfere with any technical
          limitations, safety features, content filters, watermarking,
          content provenance or latent disclosure functionalities, or
          use restrictions implemented in LTX-2.x by Licensor;

      20) To use LTX-2.x or Derivatives of LTX-2.x in any product,
          service, or application that directly competes with Licensor's
          commercial products or services, or is designed to replace or
          substitute Licensor's offerings in the market, without
          obtaining a separate commercial license from Licensor.
```

**Videos are machine generated, and must say so.** Every video this service makes is marked as
AI-generated in its file's metadata. Do not remove, disable or alter that marking, or any other
disclosure, metadata, watermark or content-provenance feature of a video (section 6 of the
licence). Wherever you publish a video or place it in any context, say plainly that it is machine
generated (Attachment A, item 5).

## SDXL base 1.0 (pictures)

SDXL base 1.0 is licensed under the CreativeML Open RAIL++-M License of Stability AI (dated July
26, 2023). SDXL base 1.0 is subject to paragraph 5 of that licence, its use-based restrictions,
and to Attachment A. By using it here you agree to comply with them, and they are an enforceable
part of these terms. They are quoted below as the licence states them; words the licence defines,
such as Model, Derivatives of the Model and Output, have the meanings it gives them. The whole
licence is at `/terms/licenses/creativeml-openrail-pp-m` on this server, and in the model's
repository, https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.

Paragraph 5 of the licence:

```text
Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
```

Attachment A of the licence:

```text
Attachment A
Use Restrictions
You agree not to use the Model or Derivatives of the Model:
In any way that violates any applicable national, federal, state, local or international law or regulation;
For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
To generate or disseminate verifiably false information and/or content with the purpose of harming others;
To generate or disseminate personal identifiable information that can be used to harm an individual;
To defame, disparage or otherwise harass others;
For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
To provide medical advice and medical results interpretation;
To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).
```

## EmbeddingGemma 2 (embeddings)

EmbeddingGemma 2 is licensed under Apache 2.0. Its model card says:

```text
Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
```

By using EmbeddingGemma 2 here you agree to follow the Gemma Prohibited Use Policy:
https://ai.google.dev/gemma/prohibited_use_policy

## Hy4-flash

Hy4-flash is Tencent's model, licensed under the Tencent Hy License, Version 1.0. That licence is
derived from the Apache License 2.0, with additional commercial terms for large Model-as-a-Service
("MaaS") businesses in its Section 10, quoted below as the licence states it. The whole licence is
at `/terms/licenses/tencent-hy-1.0` on this server, and in the model's repository, https://huggingface.co/keptn/hy4-flash-iqr-v4.
The container served here is keptn's quantised form of tencent/Hy4-flash; its NOTICE says the
quantisation changed the weights.

```text
10. Additional Commercial Terms.

   10.1 MaaS. "MaaS" means giving third parties remote access to model
        inference or fine-tuning (e.g., via API), allowing them to
        submit inputs or specify parameters or training data. This
        excludes (i) end-user products with model capabilities confined
        to specific features or workflows, without general-purpose
        model access, or (ii) merely relaying requests to an
        independently operated third-party model service.

   10.2 Separate License Required. Notwithstanding Sections 2 and 3,
        and except as provided in Section 10.3, if You operate a MaaS
        business, You must obtain a separate written license to use the
        Work or any Derivative Works from Licensor (including by email),
        in either of the following cases:

        (a) if Your aggregate worldwide gross revenue exceeds
            US$50 million (or the equivalent in other currencies) in
            any consecutive 12-month period, whether or not such
            revenue is derived from the Work or any Derivative Works,
            before using (or continuing to use) the Work or any
            Derivative Works; or

        (b) before using the Work or any Derivative Works in or for any
            product or service provided by You that has more than
            100 million monthly active users in any calendar month.

   10.3 Internal Use. Section 10.2 does not apply to Your use of the
        Work or any Derivative Works solely for internal purposes,
        provided that neither the Work, any Derivative Works, nor their
        outputs are made available to third parties.

   10.4 Contact. For licensing requests or questions, please contact
        hy_partners@tencent.com.

END OF TERMS AND CONDITIONS
```

What it means in practice:

- **A separate licence for large MaaS businesses.** If the operator runs a MaaS business (giving
  third parties general-purpose remote access to model inference, as serving this server's API to
  other people can be), it needs a separate written licence from Tencent before using the model
  when its worldwide gross revenue exceeds US$50 million in any 12 months, or before using the
  model in a product or service with more than 100 million monthly active users. Use solely for
  internal purposes, with neither the model nor its outputs made available to third parties, is
  excepted. Tencent's contact for licences is hy_partners@tencent.com.
- **The notices.** Copies of the model or of derivative works keep the licence, the copyright and
  attribution notices, and a readable copy of the NOTICE file's attribution notices.

By using Hy4-flash here you agree to its licence.

## Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is licensed under the Qwen Community License 1.0 (Copyright (c) 2026 Qwen). It
lets anyone use, copy, modify, distribute, deploy and host the model, subject to two conditions,
quoted below as the licence states them. The whole licence is at
`/terms/licenses/qwen-community-1.0` on this server, and in the model's repository,
https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4.

The conditions and the licence's definitions:

```text
1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties.
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.
```

What they mean in practice:

- **A separate licence for some commercial use.** If the operator, or any of its affiliates, runs
  a Model as a Service or AI Work Assistant business, it needs a separate licence from Qwen before
  it uses the model for any commercial purpose. Serving the model to other people through this
  server's API can be Model as a Service under the definition above. Qwen's contact for licences
  is model-business@notice.qwencloud.com.
- **The model's name on large products.** A commercial product or service built on the model
  with more than 100,000,000 monthly active users or US$ 20,000,000 monthly revenue must show the
  model's name prominently in its user interface.
- **The notice.** Copies of the model, or substantial portions of it, keep the licence's copyright
  and permission notice.

The licence also says: "THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS,
AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY." By using
Qwen3.8-Flash-Next here you agree to do so.

## Qwen-Image 2.1 (pictures)

Qwen-Image 2.1 is licensed under the Qwen RESEARCH LICENSE AGREEMENT of Hangzhou Tongyi Laboratory
Technology Co., Ltd. (release date September 20, 2026). It allows use for non-commercial purposes
only, which it defines as research or evaluation. The whole licence is at
`/terms/licenses/qwen-research` on this server, and in the model's repository. Qwen is licensed
under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology
Co., Ltd. All Rights Reserved.

Section 2 of the licence:

```text
a. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under our intellectual property or other rights owned by us embodied in the Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY.
b. You shall not use the Materials for any commercial purpose without obtaining a separate commercial license from us. If you wish to use the Materials commercially, you shall request a license from us at model-business@notice.qwencloud.com.
```

By using Qwen-Image 2.1 here you agree to use it for research or evaluation only, and to follow
section 4 of the licence ("Rules of use"), at `/terms/licenses/qwen-research`.
