import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Logo } from "../components/ui";

// Privacy policy, terms of service and support: public pages, linked from the landing, Billing and Stripe.
// Any change to what the server collects or who processes it must be reflected here.

const OWNER = "Iker Leñena Ibarrola";  // the person who runs EyeForge (data controller and seller)
const CONTACT = "contact@eyeforge.app";
const UPDATED = "October 9, 2026";

function Page({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-line">
        <div className="mx-auto flex h-16 max-w-3xl items-center justify-between px-4">
          <Link to="/" className="flex items-center gap-2 font-semibold"><Logo /> EyeForge</Link>
          <nav className="flex gap-4 text-sm text-muted">
            <Link to="/privacy">Privacy</Link><Link to="/terms">Terms</Link><Link to="/support">Support</Link>
          </nav>
        </div>
      </header>
      <article className="mx-auto max-w-3xl space-y-4 px-4 py-10 text-[15px] leading-relaxed text-ink
                          [&_h2]:pt-4 [&_h2]:text-lg [&_h2]:font-semibold [&_ul]:list-disc [&_ul]:space-y-1 [&_ul]:pl-5
                          [&_a]:text-brand-600 [&_a]:underline">
        <h1 className="text-3xl font-semibold tracking-tight">{title}</h1>
        <p className="text-sm text-muted">Last updated: {UPDATED}</p>
        {children}
      </article>
    </div>
  );
}

const Mail = () => <a href={`mailto:${CONTACT}`}>{CONTACT}</a>;

export function Privacy() {
  return (
    <Page title="Privacy policy">
      <p>
        EyeForge (eyeforge.app) is run by {OWNER}, Spain, who is responsible for your personal data. For anything
        about your data, write to <Mail />.
      </p>

      <h2>What we collect</h2>
      <ul>
        <li><b>Your account:</b> the name, email and account id that Google or GitHub give us when you sign in.</li>
        <li><b>What you give us to build a model:</b> your photos, what you write in the chat, your datasets and
          the models we train for you.</li>
        <li><b>Billing:</b> your balance, what each model, chat or labeling cost, your top-ups, and the brand and
          last 4 digits of your card. Your card number is handled by Stripe and never reaches our server.</li>
        <li><b>Technical data:</b> your IP address and requests in the server logs, used for security.</li>
      </ul>

      <h2>Cookies</h2>
      <p>
        Only the ones the site needs to work: your session (30 days), a login check (10 minutes) and a browser id
        so you only see your own work. No analytics, no ads, no tracking across sites.
      </p>

      <h2>Why we use it</h2>
      <ul>
        <li>To provide the service you asked for: train, label and deliver your models (contract).</li>
        <li>To charge and keep accounting and tax records (legal obligation).</li>
        <li>To keep the service secure and stop abuse (legitimate interest).</li>
      </ul>
      <p>
        Your photos and models are yours. We use them only to build your own models: we do not sell them, share
        them with other users or use them to train models for anyone else.
      </p>

      <h2>Who processes it for us</h2>
      <ul>
        <li><b>Nebius</b> (Netherlands, servers in Finland): cloud GPUs that train your models, storage of each
          job, and the AI models of the chat and the labeling (Token Factory).</li>
        <li><b>Hetzner</b> (Germany, servers in the EU): the server that runs the website and keeps the accounts.</li>
        <li><b>Cloudflare</b>: delivers and protects the website.</li>
        <li><b>Stripe</b>: card payments.</li>
        <li><b>Google</b> and <b>GitHub</b>: sign in, if you use them.</li>
        <li><b>Tavily</b>: web searches the chat may run to learn about what you want to detect (only the search
          text is sent, never your photos).</li>
      </ul>
      <p>
        Some of them (Cloudflare, Stripe, Google, GitHub, Tavily) may process data outside the European Union, under
        the European Commission's standard contractual clauses or the EU-US Data Privacy Framework.
      </p>

      <h2>How long we keep it</h2>
      <ul>
        <li>Your photos, datasets and models: until you delete them or close your account.</li>
        <li>Your account: until you delete it (Billing, Delete account).</li>
        <li>Payment and billing records: as long as Spanish tax and commercial law requires (up to 6 years).</li>
        <li>Backups: deleted data can remain in backups for up to 7 days.</li>
      </ul>

      <h2>Your rights</h2>
      <p>
        You can delete your account and everything in it yourself, in Billing: it is deleted here and in the cloud
        at once (only backups keep it, for up to 7 days). You can also ask to access, correct, delete or take a copy
        of your data, or object to its use, by writing to{" "}
        <Mail /> from your account's email. We answer within a month. If you are not satisfied, you can complain to
        the Spanish data protection authority (<a href="https://www.aepd.es">aepd.es</a>) or the one in your country.
      </p>

      <h2>Photos of people</h2>
      <p>
        If your photos show people who can be recognized, you are responsible for having the right to use them.
        EyeForge is not meant to identify people.
      </p>

      <h2>Age</h2>
      <p>EyeForge is not meant for people under 16.</p>
    </Page>
  );
}

export function Terms() {
  return (
    <Page title="Terms of service">
      <p>
        These terms cover the EyeForge service at eyeforge.app, run by {OWNER}, Spain (contact: <Mail />). By using
        it you accept them. EyeForge's source code is open under the AGPL-3.0 license; these terms are only about
        the service we run.
      </p>

      <h2>The service</h2>
      <p>
        You describe what a camera should see and give us some photos; EyeForge trains a small vision model for your
        device using cloud GPUs and AI models. Results depend on your photos and your task: we cannot promise a given
        accuracy. EyeForge is new and changes often.
      </p>

      <h2>Your account</h2>
      <p>
        You sign in with Google or GitHub. Keep that account safe: what happens in your EyeForge account is your
        responsibility.
      </p>

      <h2>Balance and prices</h2>
      <ul>
        <li>You top up a balance in US dollars by card and pay as you use it. Each model, chat or labeling costs what
          it used (cloud GPU time and AI model tokens) at twice what it costs us, charged as it runs and listed in
          Billing. The prices shown before you start are estimates.</li>
        <li>Prices include VAT.</li>
        <li>Your balance does not expire and cannot be transferred.</li>
        <li>Auto top-up, if you turn it on, charges your saved card the amount you chose when your balance falls
          below your threshold, at most once every 10 minutes. You can turn it off at any time in Billing.</li>
      </ul>

      <h2>Refunds</h2>
      <ul>
        <li>You can ask for a refund of the unused part of a top-up within 14 days of buying it.</li>
        <li>If you delete your account (in Billing), what is left of your paid balance goes back to your card at
          once.</li>
        <li>If a model fails because of a problem on our side, write to us and we return what it cost.</li>
        <li>What you have already used cannot be refunded: it is spent on cloud computing as soon as it runs, which
          you ask for when you start it.</li>
      </ul>
      <p>Write to <Mail /> from your account's email; refunds go back to the card you paid with.</p>

      <h2>Your content and your models</h2>
      <p>
        Your photos, texts and the models we train for you are yours, and you can use the models for anything
        lawful, including commercially. You let us store and process them only to provide the service. You confirm
        you have the right to upload what you upload.
      </p>

      <h2>What you may not do</h2>
      <ul>
        <li>Use EyeForge for anything illegal, or to watch or identify people without a legal basis.</li>
        <li>Upload content you have no right to use, or that is illegal.</li>
        <li>Attack the service, get around its limits or resell access to it.</li>
      </ul>
      <p>We may suspend accounts that do. Any unused balance is refunded, unless the law requires otherwise.</p>

      <h2>Availability and liability</h2>
      <p>
        We work to keep EyeForge running but cannot promise it will always be available or free of errors. If we
        close the service, we will tell you in advance and refund your unused balance. As far as the law allows, our
        liability is limited to what you paid us in the last 12 months. Nothing in these terms limits your rights as
        a consumer.
      </p>

      <h2>Changes</h2>
      <p>
        If we change these terms, we will show it in the app at least 30 days before. If you do not agree, you can
        close your account and get your unused balance back.
      </p>

      <h2>Law</h2>
      <p>
        These terms are governed by Spanish law. If you are a consumer in the EU, you keep the protections of the law
        of your country and can go to its courts.
      </p>
    </Page>
  );
}

export function Support() {
  return (
    <Page title="Support">
      <p>
        Write to <Mail />, preferably from your account's email. Tell us what you were doing and, if it is about a
        model, its name. We usually answer within two working days.
      </p>
      <ul>
        <li><b>Payments and refunds:</b> see the <Link to="/terms">terms</Link>. Refunds go back to your card.</li>
        <li><b>Your data:</b> see the <Link to="/privacy">privacy policy</Link>.</li>
        <li><b>Judging the hackathon?</b> Accounts with an nvidia.com, nebius.com, tavily.com or devpost.com email are
          free. With another account, write to us from its email.</li>
      </ul>
    </Page>
  );
}
