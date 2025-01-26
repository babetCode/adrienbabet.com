function initializeFilter(tomlData) {
    const items = parseToml(tomlData);
    const filterGroups = document.getElementById('filter-groups');
    const allowedFilters = filterGroups.dataset.allowedFilters;
    
    // Get all categories or only allowed ones
    const categories = getCategories(items, allowedFilters ? allowedFilters.split(',') : null);
    
    createFilterUI(categories);
    displayItems(items);
    setupEventListeners(items);
  }
  
  function parseToml(tomlData) {
    if (typeof tomlData === 'object') return tomlData.bones || [];
    
    const items = [];
    const lines = tomlData.split('\n');
    let currentItem = {};
    
    lines.forEach(line => {
      line = line.trim();
      if (!line) return;
      
      if (line.startsWith('[[')) {
        if (Object.keys(currentItem).length > 0) {
          items.push(currentItem);
        }
        currentItem = {};
      } else if (line.includes('=')) {
        const [key, value] = line.split('=').map(s => s.trim());
        currentItem[key] = value.replace(/^"/, '').replace(/"$/, '');
      }
    });
    
    if (Object.keys(currentItem).length > 0) {
      items.push(currentItem);
    }
    
    return items;
  }
  
  function getCategories(items, allowedFilters = null) {
    const categories = {};
    
    items.forEach(item => {
      Object.entries(item).forEach(([key, value]) => {
        // Skip 'name' and check if the category is allowed
        if (key !== 'name' && (!allowedFilters || allowedFilters.includes(key))) {
          if (!categories[key]) {
            categories[key] = new Set();
          }
          
          if (typeof value === 'string' && value.includes(',')) {
            value.split(',').forEach(v => categories[key].add(v.trim()));
          } else {
            categories[key].add(value);
          }
        }
      });
    });
    
    return categories;
  }
  
  function createFilterUI(categories) {
    const filterGroups = document.getElementById('filter-groups');
    
    Object.entries(categories).forEach(([category, values]) => {
      const group = document.createElement('div');
      group.className = 'filter-group bg-gray-50 dark:bg-gray-800 p-4 rounded';
      
      const title = document.createElement('h3');
      title.className = 'font-semibold mb-2';
      title.textContent = category.charAt(0).toUpperCase() + category.slice(1);
      
      const checkboxGroup = document.createElement('div');
      checkboxGroup.className = 'space-y-2';
      
      Array.from(values).sort().forEach(value => {
        const label = document.createElement('label');
        label.className = 'flex items-center gap-2 text-sm';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'rounded';
        checkbox.dataset.category = category;
        checkbox.dataset.value = value;
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(value));
        checkboxGroup.appendChild(label);
      });
      
      group.appendChild(title);
      group.appendChild(checkboxGroup);
      filterGroups.appendChild(group);
    });
  }
  
  function displayItems(items, activeFilters = {}) {
    const itemsGrid = document.getElementById('items-grid');
    const itemsCount = document.getElementById('items-count');
    itemsGrid.innerHTML = '';
    
    const filteredItems = filterItems(items, activeFilters);
    
    filteredItems.forEach(item => {
      const card = document.createElement('div');
      card.className = 'bg-gray-50 dark:bg-gray-800 p-4 rounded';
      
      const name = document.createElement('h3');
      name.className = 'font-semibold mb-2';
      name.textContent = item.name;
      card.appendChild(name);
      
      Object.entries(item).forEach(([key, value]) => {
        if (key !== 'name') {
          const detail = document.createElement('p');
          detail.className = 'text-sm';
          detail.innerHTML = `<strong>${key}:</strong> ${value}`;
          card.appendChild(detail);
        }
      });
      
      itemsGrid.appendChild(card);
    });
    
    itemsCount.textContent = `${filteredItems.length} items`;
  }
  
  function filterItems(items, activeFilters) {
    if (Object.keys(activeFilters).length === 0) {
      return items;
    }
    
    return items.filter(item => {
      return Object.entries(activeFilters).every(([category, values]) => {
        if (values.length === 0) return true;
        
        const itemValue = item[category];
        if (!itemValue) return false;
        
        // Handle comma-separated values
        const itemValues = itemValue.includes(',') 
          ? itemValue.split(',').map(v => v.trim())
          : [itemValue];
        
        // OR logic within same category
        return values.some(value => itemValues.includes(value));
      });
    });
  }
  
  function setupEventListeners(items) {
    const activeFilters = {};
    
    document.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
      checkbox.addEventListener('change', () => {
        const category = checkbox.dataset.category;
        const value = checkbox.dataset.value;
        
        if (!activeFilters[category]) {
          activeFilters[category] = [];
        }
        
        if (checkbox.checked) {
          activeFilters[category].push(value);
        } else {
          const index = activeFilters[category].indexOf(value);
          if (index > -1) {
            activeFilters[category].splice(index, 1);
          }
        }
        
        displayItems(items, activeFilters);
      });
    });
    
    const sortSelect = document.getElementById('sort-select');
    sortSelect.addEventListener('change', () => {
      const sortedItems = [...items].sort((a, b) => 
        a[sortSelect.value].localeCompare(b[sortSelect.value])
      );
      displayItems(sortedItems, activeFilters);
    });
  }