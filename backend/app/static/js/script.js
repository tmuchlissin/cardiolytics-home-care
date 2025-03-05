document.addEventListener("DOMContentLoaded", function () {
  const toggleButtonSidebar = document.getElementById("toggleSidebar");
  const sidebar = document.getElementById("sidebar");
  const mainContent = document.querySelector(".main-content");

  // Periksa status sidebar dari localStorage
  const sidebarState = localStorage.getItem("sidebarState");

  if (sidebarState === "hidden") {
    sidebar.classList.add("hidden");
    mainContent.classList.remove("ml-64");
    toggleButtonSidebar.innerHTML = '<i class="fas fa-bars"></i>';
  } else {
    sidebar.classList.remove("hidden");
    mainContent.classList.add("ml-64");
    toggleButtonSidebar.innerHTML = '<i class="fas fa-times"></i>';
  }

  toggleButtonSidebar.addEventListener("click", function () {
    const computedDisplay = window.getComputedStyle(sidebar).display;
    if (computedDisplay === "none") {
      sidebar.style.display = "block";
      mainContent.classList.add("ml-64");
      toggleButtonSidebar.innerHTML = '<i class="fas fa-times"></i>';
      localStorage.setItem("sidebarState", "visible");
    } else {
      sidebar.style.display = "none";
      mainContent.classList.remove("ml-64");
      toggleButtonSidebar.innerHTML = '<i class="fas fa-bars"></i>';
      localStorage.setItem("sidebarState", "hidden");
    }
  });
});
