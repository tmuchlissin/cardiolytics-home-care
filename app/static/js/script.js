document.addEventListener("DOMContentLoaded", function () {
  const toggleButtonSidebar = document.getElementById("toggleSidebar");
  const sidebar = document.getElementById("sidebar");
  const mainContent = document.querySelector(".main-content"); // Select the main content

  // Initial setup: Sidebar visible
  sidebar.classList.remove("hidden");
  toggleButtonSidebar.innerHTML = '<i class="fas fa-times"></i>';
  mainContent.classList.add("ml-64"); // Add margin to main content

  toggleButtonSidebar.addEventListener("click", function () {
    sidebar.classList.toggle("hidden");

    // Toggle margin class on main content
    if (sidebar.classList.contains("hidden")) {
      toggleButtonSidebar.innerHTML = '<i class="fas fa-bars"></i>';
      mainContent.classList.remove("ml-64"); // Remove margin when sidebar is hidden
    } else {
      toggleButtonSidebar.innerHTML = '<i class="fas fa-times"></i>';
      mainContent.classList.add("ml-64"); // Add margin when sidebar is visible
    }
  });
});
