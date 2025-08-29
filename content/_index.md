---
title: Home
layout: hextra-home
---

<div class="flex w-full flex-row lg:flex-row gap-8 justify-center items-center">
  <div class="flex flex-col items-center gap-2">
    <img src="/images/headshotnobg.png" alt="Adrien Babet Photo" class="headshot-photo w-36 h-36 md:w-44 md:h-44 lg:w-48 lg:h-48 rounded-full object-cover" />
    <div class="block lg:hidden">
        <div class="mt-2">{{< card link="https://adrienbabet.com/pdfs/CV.pdf" title="CV" icon="document-text">}}</div>
        <div class="mt-2">{{< card link="https://www.linkedin.com/in/adrien-babet-37bb29301/" title="LinkedIn" icon="linkedin">}}</div>
  </div>
  </div>

  
  <div>
  <div class="hx-mt-6 hx-mb-6">
  {{< hextra/hero-headline >}}
  Adrien Babet
  {{< /hextra/hero-headline >}}
  </div>

  <div class="hx-mb-12">
  {{< hextra/hero-subtitle >}}
  Kinesiology @ Montana State University.&nbsp;<br class="sm:hx-block hx-hidden"><br class="sm:hx-block hx-hidden">I love biomechanics, math, programming, and snowboarding.
  {{< /hextra/hero-subtitle >}}
  </div>
  </div>

  <div class="hidden lg:block">
        <div class="mt-2">{{< card link="https://adrienbabet.com/pdfs/CV.pdf" title="CV" icon="document-text">}}</div>
        <div class="mt-2">{{< card link="https://www.linkedin.com/in/adrien-babet-37bb29301/" title="LinkedIn" icon="linkedin">}}</div>
  </div>
</div>

<div class="hx-mt-6 w-full flex flex-col md:flex-row justify-center gap-4 md:gap-8 max-w-4xl mx-auto">
  <div class="w-full md:w-auto md:min-w-[22rem] lg:min-w-[30rem]">
    {{< hextra/feature-card
      title="Projects"
      link="/projects"
      class="hx-aspect-auto md:hx-aspect-[1.1/1] max-md:hx-min-h-[340px]"
      image="images/bme/AdrienBabet1.jpg"
      imageClass="hx-w-[110%] sm:hx-w-[110%] dark:hx-opacity-80"
      style="background: radial-gradient(ellipse at 50% 80%,rgba(224, 215, 43, 0.15),hsla(0,0%,100%,0));"
    >}}
  </div>
  <div class="w-full md:w-auto md:min-w-[22rem] lg:min-w-[30rem]">
    {{< hextra/feature-card
      title="About"
      link="/about"
      class="hx-aspect-auto md:hx-aspect-[1.1/1] max-md:hx-min-h-[340px]"
      image="/images/dragon.jpg"
      imageClass="hx-w-[110%] sm:hx-w-[110%] dark:hx-opacity-80"
      style="background: radial-gradient(ellipse at 50% 80%,rgba(48, 93, 206, 0.15),hsla(0,0%,100%,0));"
    >}}
  </div>
</div>

<style>
  .hextra-card {
    max-width: 9.375em;
    width: 9.375em;
  }
  .headshot-photo {
    box-shadow: 0 4px 5px rgba(0, 0, 0, 0.15);
  }
  .dark .headshot-photo {
    box-shadow: 0 4px 5px rgba(255, 255, 255, 0.15);
  }
</style>
