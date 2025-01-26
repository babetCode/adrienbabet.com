---
draft: false
title: Contact
---

{{< rawhtml >}}
<form name="contact" netlify class="space-y-4 mx-auto">
  <input type="hidden" name="form-name" value="contact">
  
  <div>
    <label for="name">Name</label>
    <input type="text" id="name" name="name" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
  </div>

  <div>
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
  </div>

  <div>
    <label for="message">Message</label>
    <textarea id="message" name="message" required class="mt-1 block w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500" rows="4"></textarea>
  </div>

  <button type="submit" class="w-full mt-4 py-2 px-4 bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
    Send
  </button>
</form>
{{< /rawhtml >}}