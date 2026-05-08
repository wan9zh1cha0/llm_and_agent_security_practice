# Preliminary[^1]
C2PA stands for 'The Coalition for Content Provenance and Authenticity'.

C2PA manifest consists of the assertion, claim and claim signature concerning the content, and is stored at the asset's Content Credential. 

* assertion: may include metadata (e.g., camera information such as maker or lens); actions performed on the asset (e.g., clipping, color correction); thumbnail of the asset or its ingredients; content bindings (e.g., cryptographic hashes).

* claim: collect the (JUMBF URI) references of the assertions and the signature, and possibly redaction records. 

* claim signature: cryptographically sign to the claim. 

One image may include multiple manifests, which are saved in the manifest store. When an image is redacted, a new manifest may be added to the store. 

# Introduction
This project is for C2PA detection and testing. The goal is to check whether C2PA Content Credentials remain readable after common platform-side image processing operations, including cropping, resizing, and PNG-to-JPEG conversion. 

C2PA Content Credential may be removed by minor editing if the processing pipeline does not explicitly preserve or regenerate the manifest. 

! This project is not intended for production use. 

References:

[^1]: https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
